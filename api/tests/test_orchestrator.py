# api/tests/test_orchestrator.py
import asyncio
import os
os.environ.setdefault("DATABASE_URL", "sqlite+aiosqlite:///:memory:")
os.environ.setdefault("MINIO_ENDPOINT", "http://localhost:9000")
os.environ.setdefault("MINIO_BUCKET", "test-furnitur")
os.environ.setdefault("MINIO_ROOT_USER", "minioadmin")
os.environ.setdefault("MINIO_ROOT_PASSWORD", "changeme")

import pytest
from db import init_db, Job, SessionLocal
from orchestrator import run_pipeline, subscribe_job, unsubscribe_job
from tests.conftest import MockBackend
from unittest.mock import MagicMock, patch


@pytest.fixture(autouse=True)
async def setup_db():
    await init_db()


@pytest.mark.asyncio
async def test_run_pipeline_url_job_completes():
    """URL job reaches status=completed and pushes job_completed SSE event."""
    from db import SessionLocal
    import uuid

    job_id = str(uuid.uuid4())
    async with SessionLocal() as session:
        job = Job(id=job_id, input_type="url", input_url="https://example.com/sofa")
        session.add(job)
        await session.commit()

    queue: asyncio.Queue = asyncio.Queue()
    subscribe_job(job_id, queue)

    # Patch upload_bytes so we don't need real MinIO in this unit test
    with patch("orchestrator.upload_bytes", return_value="fake-key"), \
         patch("orchestrator.presigned_url", return_value="http://minio/fake"):
        await run_pipeline(job_id, MockBackend())

    async with SessionLocal() as session:
        job = await session.get(Job, job_id)
        assert job.status == "completed"
        assert job.render_front_s3_key is not None

    # Collect all SSE events
    events = []
    while not queue.empty():
        events.append(queue.get_nowait())

    event_types = [e["type"] for e in events]
    assert "job_completed" in event_types
    assert event_types.count("stage_started") >= 7   # one per stage
    assert event_types.count("stage_completed") >= 7


@pytest.mark.asyncio
async def test_run_pipeline_image_job_skips_scrape():
    """Image upload job skips S0/S1/S2, still reaches completed."""
    import uuid
    from unittest.mock import AsyncMock

    job_id = str(uuid.uuid4())
    async with SessionLocal() as session:
        job = Job(id=job_id, input_type="image", input_s3_key=f"jobs/{job_id}/input.png")
        session.add(job)
        await session.commit()

    with patch("orchestrator.upload_bytes", return_value="fake-key"), \
         patch("orchestrator.presigned_url", return_value="http://minio/fake"), \
         patch("orchestrator._download_input_image", return_value=b"fake_bytes"):
        await run_pipeline(job_id, MockBackend())

    async with SessionLocal() as session:
        job = await session.get(Job, job_id)
        assert job.status == "completed"


@pytest.mark.asyncio
async def test_run_pipeline_failure_updates_status():
    """If a backend call raises, status becomes failed."""
    import uuid

    class FailingBackend(MockBackend):
        async def scrape(self, page_url, job_id):
            raise RuntimeError("Modal unavailable")

    job_id = str(uuid.uuid4())
    async with SessionLocal() as session:
        job = Job(id=job_id, input_type="url", input_url="https://example.com/sofa")
        session.add(job)
        await session.commit()

    queue: asyncio.Queue = asyncio.Queue()
    subscribe_job(job_id, queue)

    with patch("orchestrator.upload_bytes", return_value="fake-key"), \
         patch("orchestrator.presigned_url", return_value="http://minio/fake"):
        await run_pipeline(job_id, FailingBackend())

    async with SessionLocal() as session:
        job = await session.get(Job, job_id)
        assert job.status == "failed"
        assert "Modal unavailable" in job.error_message

    events = []
    while not queue.empty():
        events.append(queue.get_nowait())
    assert any(e["type"] == "job_failed" for e in events)
