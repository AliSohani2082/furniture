# api/compute/modal_backend.py
"""
Modal implementation of ComputeBackend.
Wraps all .remote() calls in run_in_executor to avoid blocking FastAPI's event loop.

Stage imports are deferred to each method body so that importing this module
at the top level of main.py does not pull in Modal/stages at import time.
"""
from __future__ import annotations

import asyncio
from functools import partial

import modal

from images import ARTIFACTS_DIR, ARTIFACTS_VOLUME, BASE_IMAGE, app as modal_app

# A small Modal function for reading artifacts out of the Modal Volume.
# Defined here rather than importing from main.py to keep this module self-contained.
@modal_app.function(
    image=BASE_IMAGE,
    volumes={"/vol": ARTIFACTS_VOLUME},
    timeout=120,
)
def _fetch_volume_files(job_id: str, rels: list[str]) -> dict[str, bytes]:
    ARTIFACTS_VOLUME.reload()
    return {
        rel: (ARTIFACTS_DIR / rel).read_bytes()
        for rel in rels
        if rel and (ARTIFACTS_DIR / rel).exists()
    }


@modal_app.function(
    image=BASE_IMAGE,
    volumes={"/vol": ARTIFACTS_VOLUME},
    timeout=60,
)
def _write_image_to_volume(image_bytes: bytes, job_id: str) -> str:
    """Write uploaded image bytes into Modal Volume, return the rel path."""
    from io import BytesIO
    from PIL import Image
    from pathlib import Path

    job_dir = ARTIFACTS_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    img = Image.open(BytesIO(image_bytes)).convert("RGB")
    dest = job_dir / "raw_input.png"
    img.save(dest)
    ARTIFACTS_VOLUME.commit()
    return str(dest.relative_to(ARTIFACTS_DIR))


async def _run_in_executor(fn, *args):
    """Run a blocking Modal .remote() call without blocking the event loop."""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, fn, *args)


class ModalBackend:
    """Implements ComputeBackend using the Modal SDK."""

    async def scrape(self, page_url: str, job_id: str) -> dict:
        from stages.s0_scrape import scrape_page
        return await _run_in_executor(scrape_page.remote, page_url, job_id)

    async def analyze(self, scrape_result: dict, job_id: str) -> dict:
        from stages.s1_intelligence import analyze_page
        return await _run_in_executor(analyze_page.remote, scrape_result, job_id)

    async def crop(self, job_id: str, intel_result: dict) -> dict:
        from stages.s2_crop import Cropper
        return await _run_in_executor(Cropper().crop_all.remote, job_id, intel_result)

    async def prepare_image(self, image_bytes: bytes, job_id: str) -> tuple[dict, dict]:
        """
        For uploaded images: skip S0/S1/S2. Write image to Modal Volume,
        return synthetic (intel_result, crop_result) that S3 expects.
        """
        raw_rel = await _run_in_executor(_write_image_to_volume.remote, image_bytes, job_id)

        intel_result = {
            "furniture_category": "furniture",
            "material_hints": [],
            "dimensions_mm": None,
            "dimensions_source": "absent",
            "view_classifications": [
                {"url": "local:upload", "view": "front", "confidence": 1.0, "is_product_isolated": True}
            ],
            "reconstruction_candidates": ["local:upload"],
            "texture_candidates": ["local:upload"],
        }
        crop_result = {
            "job_id": job_id,
            "crops": [
                {
                    "index": 0,
                    "source_url": "local:upload",
                    "view_label": "front",
                    "crop_rel": raw_rel,
                    "mask_rel": None,
                    "fallback": True,
                }
            ],
        }
        return intel_result, crop_result

    async def reconstruct(
        self, job_id: str, crop_result: dict, intel_result: dict, quality: str
    ) -> dict:
        from stages.s3_reconstruct import InstantMeshGenerator
        fn = partial(InstantMeshGenerator().generate.remote, job_id, crop_result, intel_result, quality=quality)
        return await _run_in_executor(fn)

    async def scale(self, job_id: str, reconstruct_result: dict, intel_result: dict) -> dict:
        from stages.s4_scale import scale_mesh
        dims = intel_result.get("dimensions_mm")
        source = intel_result.get("dimensions_source", "absent")
        return await _run_in_executor(scale_mesh.remote, job_id, reconstruct_result, dims, source)

    async def texture(
        self, job_id: str, scale_result: dict, crop_result: dict, intel_result: dict
    ) -> dict:
        from stages.s5_texture import TextureFuser
        fn = partial(TextureFuser().fuse.remote, job_id, scale_result, crop_result, intel_result)
        return await _run_in_executor(fn)

    async def render(self, job_id: str, texture_result: dict) -> dict:
        from stages.s6_render import render_views
        return await _run_in_executor(render_views.remote, job_id, texture_result)

    async def fetch_files(self, job_id: str, rels: list[str]) -> dict[str, bytes]:
        return await _run_in_executor(_fetch_volume_files.remote, job_id, [r for r in rels if r])
