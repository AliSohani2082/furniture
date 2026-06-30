# api/main.py
from __future__ import annotations

import asyncio
import json
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone

from fastapi import BackgroundTasks, Depends, File, Form, HTTPException, UploadFile
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from compute import ComputeBackend
from compute.modal_backend import ModalBackend
from db import Job, get_session, init_db
from orchestrator import run_pipeline, subscribe_job, unsubscribe_job
from storage import ensure_bucket, presigned_url, upload_bytes

# Default backend — replaced in tests via `main.backend = mock_backend`.
backend: ComputeBackend = ModalBackend()


@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()
    ensure_bucket()
    yield


app = FastAPI(title="Furniture Pipeline API", lifespan=lifespan)


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

@app.get("/health")
async def health():
    return {"status": "ok"}


# ---------------------------------------------------------------------------
# Job creation
# ---------------------------------------------------------------------------

@app.post("/jobs", status_code=201)
async def create_job(
    background_tasks: BackgroundTasks,
    session: AsyncSession = Depends(get_session),
    url: str | None = Form(None),
    image: UploadFile | None = File(None),
):
    if url is None and image is None:
        raise HTTPException(422, "Provide either 'url' form field or 'image' file")

    job = Job(id=str(uuid.uuid4()))

    if url:
        job.input_type = "url"
        job.input_url = url
    else:
        contents = await image.read()
        suffix = image.filename.rsplit(".", 1)[-1] if image.filename and "." in image.filename else "png"
        s3_key = f"jobs/{job.id}/input.{suffix}"
        upload_bytes(s3_key, contents, image.content_type or "image/png")
        job.input_type = "image"
        job.input_s3_key = s3_key

    session.add(job)
    await session.commit()
    background_tasks.add_task(run_pipeline, job.id, backend)
    return {"id": job.id, "status": job.status}


# ---------------------------------------------------------------------------
# Job listing and detail
# ---------------------------------------------------------------------------

@app.get("/jobs")
async def list_jobs(session: AsyncSession = Depends(get_session)):
    result = await session.execute(select(Job).order_by(Job.created_at.desc()))
    jobs = result.scalars().all()
    return [_job_summary(j) for j in jobs]


@app.get("/jobs/{job_id}")
async def get_job(job_id: str, session: AsyncSession = Depends(get_session)):
    job = await session.get(Job, job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    return _job_detail(job)


@app.delete("/jobs/{job_id}")
async def cancel_job(job_id: str, session: AsyncSession = Depends(get_session)):
    job = await session.get(Job, job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    await session.execute(
        update(Job).where(Job.id == job_id).values(status="cancelled", updated_at=datetime.now(timezone.utc))
    )
    await session.commit()
    return {"id": job_id, "status": "cancelled"}


# ---------------------------------------------------------------------------
# SSE stream
# ---------------------------------------------------------------------------

@app.get("/jobs/{job_id}/stream")
async def stream_job(job_id: str, session: AsyncSession = Depends(get_session)):
    job = await session.get(Job, job_id)
    if not job:
        raise HTTPException(404, "Job not found")

    queue: asyncio.Queue = asyncio.Queue()
    subscribe_job(job_id, queue)

    async def event_generator():
        try:
            while True:
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=30.0)
                    yield f"event: {event['type']}\ndata: {json.dumps(event['data'])}\n\n"
                    if event["type"] in ("job_completed", "job_failed"):
                        break
                except asyncio.TimeoutError:
                    yield ": keepalive\n\n"
        finally:
            unsubscribe_job(job_id, queue)

    return StreamingResponse(event_generator(), media_type="text/event-stream")


# ---------------------------------------------------------------------------
# Serialization helpers
# ---------------------------------------------------------------------------

def _maybe_url(key: str | None) -> str | None:
    return presigned_url(key) if key else None


def _maybe_urls(keys: list[str] | None) -> list[str]:
    return [presigned_url(k) for k in keys] if keys else []


def _job_summary(job: Job) -> dict:
    return {
        "id": job.id,
        "status": job.status,
        "input_type": job.input_type,
        "input_url": job.input_url,
        "furniture_category": job.furniture_category,
        "first_crop_url": _maybe_url(job.crop_s3_keys[0]) if job.crop_s3_keys else None,
        "created_at": job.created_at.isoformat(),
    }


def _job_detail(job: Job) -> dict:
    return {
        "id": job.id,
        "status": job.status,
        "current_stage": job.current_stage,
        "error_message": job.error_message,
        "input_type": job.input_type,
        "input_url": job.input_url,
        "furniture_category": job.furniture_category,
        "dimensions_mm": job.dimensions_mm,
        "stage_timings": job.stage_timings,
        "crop_urls": _maybe_urls(job.crop_s3_keys),
        "mask_urls": _maybe_urls(job.mask_s3_keys),
        "mesh_glb_url": _maybe_url(job.mesh_glb_s3_key),
        "mesh_scaled_glb_url": _maybe_url(job.mesh_scaled_glb_s3_key),
        "mesh_textured_glb_url": _maybe_url(job.mesh_textured_glb_s3_key),
        "texture_atlas_url": _maybe_url(job.texture_atlas_s3_key),
        "uv_map_url": _maybe_url(job.uv_map_s3_key),
        "render_front_url": _maybe_url(job.render_front_s3_key),
        "render_side_url": _maybe_url(job.render_side_s3_key),
        "render_top_url": _maybe_url(job.render_top_s3_key),
        "render_angled_url": _maybe_url(job.render_angled_s3_key),
        "created_at": job.created_at.isoformat(),
    }
