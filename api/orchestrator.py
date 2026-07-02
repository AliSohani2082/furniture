# api/orchestrator.py
"""
Orchestrates the full Pipeline E run for a single job.
Calls each stage via the ComputeBackend Protocol, uploads artifacts to MinIO,
updates PostgreSQL, and pushes SSE events to subscribed browser clients.
"""
from __future__ import annotations

import asyncio
import io
from collections import defaultdict
from datetime import datetime, timezone

import boto3
from botocore.client import Config
from sqlalchemy import update

from compute import ComputeBackend
from db import Job, SessionLocal
from storage import upload_bytes, presigned_url, MINIO_ENDPOINT, MINIO_BUCKET, MINIO_ROOT_USER, MINIO_ROOT_PASSWORD

# ---------------------------------------------------------------------------
# SSE subscription registry
# ---------------------------------------------------------------------------

_subscribers: dict[str, list[asyncio.Queue]] = defaultdict(list)


def subscribe_job(job_id: str, queue: asyncio.Queue) -> None:
    _subscribers[job_id].append(queue)


def unsubscribe_job(job_id: str, queue: asyncio.Queue) -> None:
    if queue in _subscribers.get(job_id, []):
        _subscribers[job_id].remove(queue)


def _push(job_id: str, event_type: str, data: dict) -> None:
    for q in list(_subscribers.get(job_id, [])):
        q.put_nowait({"type": event_type, "data": data})


# ---------------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------------

async def _update_job(job_id: str, **kwargs) -> None:
    kwargs["updated_at"] = datetime.now(timezone.utc)
    async with SessionLocal() as session:
        await session.execute(update(Job).where(Job.id == job_id).values(**kwargs))
        await session.commit()


# ---------------------------------------------------------------------------
# Stage labels for SSE events
# ---------------------------------------------------------------------------

STAGES = [
    ("s0_scrape",       0, "Scraping product page"),
    ("s1_intelligence", 1, "Analysing page content"),
    ("s2_crop",         2, "Cropping furniture images"),
    ("s3_reconstruct",  3, "Building 3D mesh"),
    ("s4_scale",        4, "Applying real-world dimensions"),
    ("s5_texture",      5, "Fusing textures"),
    ("s6_render",       6, "Rendering views"),
]


def _push_started(job_id: str, stage_name: str, stage_index: int, label: str) -> None:
    _push(job_id, "stage_started", {
        "stage": stage_name,
        "stage_index": stage_index,
        "label": label,
    })


def _push_completed(job_id: str, stage_name: str, stage_index: int, assets: dict) -> None:
    _push(job_id, "stage_completed", {
        "stage": stage_name,
        "stage_index": stage_index,
        "assets": assets,
    })


# ---------------------------------------------------------------------------
# Input image download helper (for image-upload jobs)
# ---------------------------------------------------------------------------

def _download_input_image(s3_key: str) -> bytes:
    client = boto3.client(
        "s3",
        endpoint_url=MINIO_ENDPOINT,
        aws_access_key_id=MINIO_ROOT_USER,
        aws_secret_access_key=MINIO_ROOT_PASSWORD,
        config=Config(signature_version="s3v4"),
        region_name="us-east-1",
    )
    buf = io.BytesIO()
    client.download_fileobj(MINIO_BUCKET, s3_key, buf)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------

async def run_pipeline(job_id: str, backend: ComputeBackend) -> None:
    """
    Run the full Pipeline E for a job. Called as a FastAPI background task.
    Updates PostgreSQL and pushes SSE events at each stage boundary.
    """
    async with SessionLocal() as session:
        job = await session.get(Job, job_id)
        input_type = job.input_type
        input_url = job.input_url
        input_s3_key = job.input_s3_key

    try:
        await _update_job(job_id, status="running")

        if input_type == "url":
            # --- S0: Scrape ---
            _push_started(job_id, *STAGES[0])
            scrape_result = await backend.scrape(input_url, job_id)
            await _update_job(job_id, current_stage=0)
            _push_completed(job_id, STAGES[0][0], STAGES[0][1], {})

            # --- S1: Intelligence ---
            _push_started(job_id, *STAGES[1])
            intel_result = await backend.analyze(scrape_result, job_id)
            await _update_job(
                job_id,
                current_stage=1,
                furniture_category=intel_result.get("furniture_category"),
                dimensions_mm=intel_result.get("dimensions_mm"),
            )
            _push_completed(job_id, STAGES[1][0], STAGES[1][1], {
                "furniture_category": intel_result.get("furniture_category"),
                "dimensions_mm": intel_result.get("dimensions_mm"),
            })

            # --- S2: Crop ---
            _push_started(job_id, *STAGES[2])
            crop_result = await backend.crop(job_id, intel_result)
            crop_keys, mask_keys, crop_urls, mask_urls = await _upload_crops(job_id, crop_result, backend)
            await _update_job(job_id, current_stage=2, crop_s3_keys=crop_keys, mask_s3_keys=mask_keys)
            _push_completed(job_id, STAGES[2][0], STAGES[2][1], {"crops": crop_urls, "masks": mask_urls})

        else:
            # Image upload: skip S0, S1, S2 — use prepare_image instead
            image_bytes = _download_input_image(input_s3_key)
            _push_started(job_id, *STAGES[0])
            _push_started(job_id, *STAGES[1])
            _push_started(job_id, *STAGES[2])
            intel_result, crop_result = await backend.prepare_image(image_bytes, job_id)
            await _update_job(job_id, current_stage=2, furniture_category="furniture")
            _push_completed(job_id, STAGES[0][0], STAGES[0][1], {})
            _push_completed(job_id, STAGES[1][0], STAGES[1][1], {})
            _push_completed(job_id, STAGES[2][0], STAGES[2][1], {})

        # Check for cancellation before the post-branch stages
        async with SessionLocal() as s:
            fresh = await s.get(Job, job_id)
            if fresh.status == "cancelled":
                return

        # --- S3: Reconstruct ---
        _push_started(job_id, *STAGES[3])
        recon_result = await backend.reconstruct(job_id, crop_result, intel_result, "large")
        files = await backend.fetch_files(job_id, [recon_result["glb_rel"], recon_result.get("uv_map_rel")])
        mesh_key = upload_bytes(f"jobs/{job_id}/meshes/mesh.glb", files[recon_result["glb_rel"]], "model/gltf-binary")
        uv_key = None
        if recon_result.get("uv_map_rel") and recon_result["uv_map_rel"] in files:
            uv_key = upload_bytes(f"jobs/{job_id}/meshes/uv_map.png", files[recon_result["uv_map_rel"]], "image/png")
        await _update_job(job_id, current_stage=3, mesh_glb_s3_key=mesh_key, uv_map_s3_key=uv_key)
        _push_completed(job_id, STAGES[3][0], STAGES[3][1], {})

        # --- S4: Scale ---
        _push_started(job_id, *STAGES[4])
        scale_result = await backend.scale(job_id, recon_result, intel_result)
        files = await backend.fetch_files(job_id, [scale_result["scaled_glb_rel"]])
        scaled_key = upload_bytes(f"jobs/{job_id}/meshes/mesh_scaled.glb", files[scale_result["scaled_glb_rel"]], "model/gltf-binary")
        await _update_job(job_id, current_stage=4, mesh_scaled_glb_s3_key=scaled_key)
        _push_completed(job_id, STAGES[4][0], STAGES[4][1], {})

        # --- S5: Texture ---
        _push_started(job_id, *STAGES[5])
        texture_result = await backend.texture(job_id, scale_result, crop_result, intel_result)
        tex_rels = [texture_result["textured_glb_rel"], texture_result["texture_atlas_rel"]]
        files = await backend.fetch_files(job_id, tex_rels)
        tex_glb_key = upload_bytes(f"jobs/{job_id}/meshes/mesh_textured.glb", files[texture_result["textured_glb_rel"]], "model/gltf-binary")
        atlas_key = upload_bytes(f"jobs/{job_id}/meshes/texture_atlas.png", files[texture_result["texture_atlas_rel"]], "image/png")
        await _update_job(job_id, current_stage=5, mesh_textured_glb_s3_key=tex_glb_key, texture_atlas_s3_key=atlas_key)
        _push_completed(job_id, STAGES[5][0], STAGES[5][1], {
            "mesh_textured_glb": presigned_url(tex_glb_key),
        })

        # --- S6: Render ---
        _push_started(job_id, *STAGES[6])
        render_result = await backend.render(job_id, texture_result)
        render_rels = {v: render_result[k] for k, v in [
            ("front_rel", "front"), ("side_rel", "side"),
            ("top_rel", "top"), ("angled_rel", "angled"),
        ]}
        files = await backend.fetch_files(job_id, list(render_rels.values()))
        render_keys = {}
        render_urls = {}
        for view, rel in render_rels.items():
            key = upload_bytes(f"jobs/{job_id}/renders/{view}.png", files[rel], "image/png")
            render_keys[f"render_{view}_s3_key"] = key
            render_urls[view] = presigned_url(key)
        await _update_job(job_id, current_stage=6, **render_keys)
        _push_completed(job_id, STAGES[6][0], STAGES[6][1], render_urls)

        # Done
        await _update_job(job_id, status="completed")
        _push(job_id, "job_completed", {"job_id": job_id})

    except Exception as exc:
        await _update_job(job_id, status="failed", error_message=str(exc))
        _push(job_id, "job_failed", {"error": str(exc)})


async def _upload_crops(
    job_id: str,
    crop_result: dict,
    backend: ComputeBackend,
) -> tuple[list[str], list[str], list[str], list[str]]:
    """Fetch crop + mask files from Modal Volume and upload to MinIO. Returns (crop_keys, mask_keys, crop_urls, mask_urls)."""
    rels_to_fetch = []
    for meta in crop_result["crops"]:
        rels_to_fetch.append(meta["crop_rel"])
        if meta.get("mask_rel"):
            rels_to_fetch.append(meta["mask_rel"])

    files = await backend.fetch_files(job_id, rels_to_fetch)

    crop_keys, mask_keys, crop_urls, mask_urls = [], [], [], []
    for i, meta in enumerate(crop_result["crops"]):
        crop_rel = meta["crop_rel"]
        if crop_rel in files:
            key = upload_bytes(f"jobs/{job_id}/crops/crop_{i}.png", files[crop_rel], "image/png")
            crop_keys.append(key)
            crop_urls.append(presigned_url(key))

        mask_rel = meta.get("mask_rel")
        if mask_rel and mask_rel in files:
            key = upload_bytes(f"jobs/{job_id}/crops/mask_{i}.png", files[mask_rel], "image/png")
            mask_keys.append(key)
            mask_urls.append(presigned_url(key))

    return crop_keys, mask_keys, crop_urls, mask_urls
