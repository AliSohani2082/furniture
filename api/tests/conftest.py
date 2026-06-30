import os
os.environ["DATABASE_URL"] = "sqlite+aiosqlite:///:memory:"
os.environ["MINIO_ENDPOINT"] = "http://localhost:9000"
os.environ["MINIO_BUCKET"] = "test-furnitur"
os.environ["MINIO_ROOT_USER"] = "minioadmin"
os.environ["MINIO_ROOT_PASSWORD"] = "changeme"

import pytest
from httpx import AsyncClient, ASGITransport
from unittest.mock import patch

from compute import ComputeBackend


class MockBackend:
    """In-memory stub. Implements ComputeBackend without any Modal/network calls."""

    async def scrape(self, page_url, job_id):
        return {"source_url": page_url, "page_text": "Sofa page", "candidate_image_urls": ["http://example.com/sofa.jpg"]}

    async def analyze(self, scrape_result, job_id):
        return {
            "furniture_category": "sofa",
            "material_hints": [],
            "dimensions_mm": {"width_mm": 900, "depth_mm": 450, "height_mm": 760},
            "dimensions_source": "page_text",
            "view_classifications": [],
            "reconstruction_candidates": ["http://example.com/sofa.jpg"],
            "texture_candidates": ["http://example.com/sofa.jpg"],
        }

    async def crop(self, job_id, intel_result):
        return {"job_id": job_id, "crops": [{"index": 0, "source_url": "http://example.com/sofa.jpg", "view_label": "front", "crop_rel": f"{job_id}/crop_0.png", "mask_rel": f"{job_id}/mask_0.png", "fallback": False}]}

    async def prepare_image(self, image_bytes, job_id):
        intel = {"furniture_category": "furniture", "material_hints": [], "dimensions_mm": None, "dimensions_source": "absent", "view_classifications": [], "reconstruction_candidates": [], "texture_candidates": []}
        crop = {"job_id": job_id, "crops": [{"index": 0, "source_url": "local:upload", "view_label": "front", "crop_rel": f"{job_id}/raw_input.png", "mask_rel": None, "fallback": True}]}
        return intel, crop

    async def reconstruct(self, job_id, crop_result, intel_result, quality):
        return {"job_id": job_id, "glb_rel": f"{job_id}/mesh.glb", "obj_rel": f"{job_id}/mesh.obj", "uv_map_rel": f"{job_id}/uv_map.png"}

    async def scale(self, job_id, reconstruct_result, intel_result):
        return {"job_id": job_id, "scaled_glb_rel": f"{job_id}/mesh_scaled.glb", "scaled_obj_rel": f"{job_id}/mesh_scaled.obj", "scale_applied": True, "scale_factor": 1.2, "dimensions_mm": None}

    async def texture(self, job_id, scale_result, crop_result, intel_result):
        return {"job_id": job_id, "textured_glb_rel": f"{job_id}/mesh_textured.glb", "textured_obj_rel": f"{job_id}/mesh_textured.obj", "texture_atlas_rel": f"{job_id}/texture_atlas.png"}

    async def render(self, job_id, texture_result):
        return {"job_id": job_id, "front_rel": f"{job_id}/front.png", "side_rel": f"{job_id}/side.png", "top_rel": f"{job_id}/top.png", "angled_rel": f"{job_id}/angled.png"}

    async def fetch_files(self, job_id, rels):
        return {rel: b"fake_bytes" for rel in rels if rel}


@pytest.fixture
def mock_backend():
    return MockBackend()


@pytest.fixture
async def client(mock_backend):
    import main
    from db import Base, engine, init_db
    main.backend = mock_backend   # inject mock before app starts
    # Initialise DB (lifespan is not triggered by ASGITransport)
    await init_db()
    # Patch ensure_bucket and upload_bytes so unit tests don't need a real MinIO instance running
    with patch("main.ensure_bucket"), patch("main.upload_bytes", return_value="fake/key"):
        async with AsyncClient(transport=ASGITransport(app=main.app), base_url="http://test") as c:
            yield c
    # Teardown: drop and recreate tables so each test starts with a clean DB
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
        await conn.run_sync(Base.metadata.create_all)
