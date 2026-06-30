# Furniture Pipeline — Backend Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the FastAPI backend, PostgreSQL schema, MinIO storage wrapper, Modal compute abstraction, and Docker Compose stack for the furniture pipeline web application.

**Architecture:** FastAPI orchestrates the existing Modal pipeline stages by calling them via `.remote()`, wrapped behind a `ComputeBackend` Protocol so the stack is AWS-migration-friendly. After each stage completes, artifacts are read from the Modal Volume and uploaded to MinIO; job metadata is written to PostgreSQL; SSE pushes live updates to browsers.

**Tech Stack:** FastAPI 0.115+, SQLAlchemy 2 (async + asyncpg), boto3 (MinIO/S3), Modal SDK, PostgreSQL 16, MinIO (minio/minio Docker image), Python 3.11, pytest + httpx for tests.

## Global Constraints

- Python 3.11 everywhere (matches Modal container images)
- `modal` SDK version must match what's already installed in the project venv (check `pip show modal` — use that version in requirements.txt)
- All Modal calls must run in a thread pool executor (`asyncio.run_in_executor`) — never block the FastAPI event loop
- No changes to existing pipeline files (`stages/`, `schemas/`, `images/`, `main.py`)
- Docker build context for the api service is the **project root** (not the api/ subdirectory), so the Dockerfile can `COPY stages/ schemas/ images/` into the container
- Database URL format: `postgresql+asyncpg://furnitur:{POSTGRES_PASSWORD}@db:5432/furnitur`
- MinIO endpoint inside Docker network: `http://storage:9000`; from the host: `http://localhost:9000`

---

## File Map

**Created by this plan:**
```
docker-compose.yml
.env.example
api/
  Dockerfile
  requirements.txt
  main.py              — FastAPI app, all route handlers, lifespan
  db.py                — SQLAlchemy Job model, engine, get_session, init_db
  storage.py           — MinIO client: upload_bytes(), presigned_url(), ensure_bucket()
  orchestrator.py      — run_pipeline(), SSE subscribe/unsubscribe/_push
  compute/
    __init__.py        — ComputeBackend Protocol (the AWS-migration seam)
    modal_backend.py   — ModalBackend: wraps all .remote() calls in run_in_executor
  tests/
    __init__.py
    conftest.py        — MockBackend, test DB fixture, test MinIO fixture
    test_health.py
    test_jobs.py
    test_orchestrator.py
```

**Unchanged:** `stages/`, `schemas/`, `images/`, `main.py`, `requirements.txt` (project root)

---

## Task 1: Docker infrastructure — PostgreSQL + MinIO

**Files:**
- Create: `docker-compose.yml`
- Create: `.env.example`

**Interfaces:**
- Produces: running `db` and `storage` services, confirming the foundation works before adding application code

- [ ] **Step 1: Write docker-compose.yml (db + storage only)**

```yaml
# docker-compose.yml
services:
  db:
    image: postgres:16-alpine
    environment:
      POSTGRES_DB: furnitur
      POSTGRES_USER: furnitur
      POSTGRES_PASSWORD: ${POSTGRES_PASSWORD}
    volumes:
      - pgdata:/var/lib/postgresql/data
    ports:
      - "5432:5432"
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U furnitur -d furnitur"]
      interval: 5s
      timeout: 5s
      retries: 10

  storage:
    image: minio/minio:latest
    ports:
      - "9000:9000"
      - "9001:9001"
    command: server /data --console-address ":9001"
    environment:
      MINIO_ROOT_USER: ${MINIO_ROOT_USER}
      MINIO_ROOT_PASSWORD: ${MINIO_ROOT_PASSWORD}
    volumes:
      - miniodata:/data
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:9000/minio/health/live"]
      interval: 5s
      timeout: 5s
      retries: 10

volumes:
  pgdata:
  miniodata:
```

- [ ] **Step 2: Write .env.example**

```env
# .env.example
# PostgreSQL
POSTGRES_PASSWORD=changeme

# MinIO
MINIO_ROOT_USER=minioadmin
MINIO_ROOT_PASSWORD=changeme
MINIO_ENDPOINT=http://storage:9000
MINIO_BUCKET=furnitur-pipeline

# Modal (required — GPU pipeline runs on Modal's cloud)
MODAL_TOKEN_ID=ak-changeme
MODAL_TOKEN_SECRET=as-changeme

# OpenRouter (used by S1 Intelligence stage)
OPENROUTER_API_KEY=sk-or-changeme
```

- [ ] **Step 3: Create .env from example and start infrastructure**

```bash
cp .env.example .env
# Fill in real POSTGRES_PASSWORD and MINIO_* values (keep Modal/OpenRouter for later)
docker compose up db storage -d
```

- [ ] **Step 4: Verify both services are healthy**

```bash
docker compose ps
```

Expected output: both `db` and `storage` show `healthy` in the Status column.

```bash
docker compose exec db psql -U furnitur -d furnitur -c "SELECT 1;"
```

Expected: `?column? ---------- 1`

```bash
curl -s http://localhost:9000/minio/health/live
```

Expected: HTTP 200 (empty body).

- [ ] **Step 5: Commit**

```bash
git add docker-compose.yml .env.example
git commit -m "feat: add Docker Compose with PostgreSQL and MinIO services"
```

---

## Task 2: FastAPI app skeleton + PostgreSQL model

**Files:**
- Create: `api/Dockerfile`
- Create: `api/requirements.txt`
- Create: `api/db.py`
- Create: `api/main.py`
- Create: `api/tests/__init__.py`
- Create: `api/tests/test_health.py`
- Modify: `docker-compose.yml` (add `api` service)

**Interfaces:**
- Produces: `GET /health → {"status": "ok"}`, `Job` SQLAlchemy model with all columns, `get_session()` async dependency, `init_db()` async function

- [ ] **Step 1: Write api/requirements.txt**

```text
fastapi>=0.115.0
uvicorn[standard]>=0.30.0
sqlalchemy[asyncio]>=2.0.0
asyncpg>=0.29.0
python-multipart>=0.0.12
boto3>=1.34.0
botocore>=1.34.0
modal>=0.64.0
httpx>=0.27.0
pytest>=8.0.0
pytest-asyncio>=0.23.0
anyio>=4.0.0
```

Replace `modal>=0.64.0` with the exact version from `pip show modal` in the project venv.

- [ ] **Step 2: Write api/Dockerfile**

```dockerfile
FROM python:3.11-slim
WORKDIR /app

# Copy pipeline source (needed so Modal can register stage functions)
COPY stages/ ./stages/
COPY schemas/ ./schemas/
COPY images/ ./images/

# Install API dependencies
COPY api/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy API source
COPY api/ .

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
```

- [ ] **Step 3: Write api/db.py**

```python
# api/db.py
from __future__ import annotations

import os
import uuid
from datetime import datetime

from sqlalchemy import ARRAY, JSON, SmallInteger, Text
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

DATABASE_URL = os.environ["DATABASE_URL"]

engine = create_async_engine(DATABASE_URL, echo=False)
SessionLocal = async_sessionmaker(engine, expire_on_commit=False)


class Base(DeclarativeBase):
    pass


class Job(Base):
    __tablename__ = "jobs"

    id: Mapped[str] = mapped_column(Text, primary_key=True, default=lambda: str(uuid.uuid4()))
    created_at: Mapped[datetime] = mapped_column(default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(default=datetime.utcnow, onupdate=datetime.utcnow)

    # Input
    input_type: Mapped[str] = mapped_column(Text)        # 'url' | 'image'
    input_url: Mapped[str | None] = mapped_column(Text, nullable=True)
    input_s3_key: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Lifecycle
    status: Mapped[str] = mapped_column(Text, default="queued")
    current_stage: Mapped[int] = mapped_column(SmallInteger, default=0)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    stage_timings: Mapped[dict] = mapped_column(JSON, default=dict)

    # S1 intelligence metadata
    furniture_category: Mapped[str | None] = mapped_column(Text, nullable=True)
    dimensions_mm: Mapped[dict | None] = mapped_column(JSON, nullable=True)

    # S2 crops (parallel arrays)
    crop_s3_keys: Mapped[list[str] | None] = mapped_column(ARRAY(Text), nullable=True)
    mask_s3_keys: Mapped[list[str] | None] = mapped_column(ARRAY(Text), nullable=True)

    # S3–S5 mesh outputs
    mesh_glb_s3_key: Mapped[str | None] = mapped_column(Text, nullable=True)
    mesh_scaled_glb_s3_key: Mapped[str | None] = mapped_column(Text, nullable=True)
    mesh_textured_glb_s3_key: Mapped[str | None] = mapped_column(Text, nullable=True)
    uv_map_s3_key: Mapped[str | None] = mapped_column(Text, nullable=True)
    texture_atlas_s3_key: Mapped[str | None] = mapped_column(Text, nullable=True)

    # S6 render outputs
    render_front_s3_key: Mapped[str | None] = mapped_column(Text, nullable=True)
    render_side_s3_key: Mapped[str | None] = mapped_column(Text, nullable=True)
    render_top_s3_key: Mapped[str | None] = mapped_column(Text, nullable=True)
    render_angled_s3_key: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Modal tracking
    modal_call_id: Mapped[str | None] = mapped_column(Text, nullable=True)


async def get_session():
    async with SessionLocal() as session:
        yield session


async def init_db():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
```

- [ ] **Step 4: Write the failing test**

```python
# api/tests/__init__.py
# (empty)
```

```python
# api/tests/test_health.py
import os
os.environ.setdefault("DATABASE_URL", "sqlite+aiosqlite:///:memory:")
os.environ.setdefault("MINIO_ENDPOINT", "http://localhost:9000")
os.environ.setdefault("MINIO_BUCKET", "test")
os.environ.setdefault("MINIO_ROOT_USER", "test")
os.environ.setdefault("MINIO_ROOT_PASSWORD", "test")

import pytest
from httpx import AsyncClient, ASGITransport


@pytest.mark.asyncio
async def test_health_returns_ok():
    from main import app
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
```

- [ ] **Step 5: Run test to verify it fails**

```bash
cd api && pip install -r requirements.txt && pip install aiosqlite
pytest tests/test_health.py -v
```

Expected: `ImportError` or `ModuleNotFoundError: No module named 'main'` — the file doesn't exist yet.

- [ ] **Step 6: Write api/main.py (skeleton)**

```python
# api/main.py
from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI

from db import init_db


@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()
    yield


app = FastAPI(title="Furniture Pipeline API", lifespan=lifespan)


@app.get("/health")
async def health():
    return {"status": "ok"}
```

- [ ] **Step 7: Run test to verify it passes**

```bash
cd api && pytest tests/test_health.py -v
```

Expected: `PASSED` — `test_health_returns_ok`.

- [ ] **Step 8: Add api service to docker-compose.yml**

Add under `services:` (after `storage:` block):

```yaml
  api:
    build:
      context: .
      dockerfile: api/Dockerfile
    ports:
      - "8000:8000"
    env_file: .env
    environment:
      DATABASE_URL: postgresql+asyncpg://furnitur:${POSTGRES_PASSWORD}@db:5432/furnitur
    depends_on:
      db:
        condition: service_healthy
      storage:
        condition: service_healthy
    volumes:
      - ~/.modal:/root/.modal:ro
```

The `~/.modal` volume mount provides Modal credentials from the host into the container. Run `modal token set --token-id ... --token-secret ...` on the host first to create `~/.modal/`.

- [ ] **Step 9: Build and smoke-test the api service**

```bash
docker compose up api --build -d
curl http://localhost:8000/health
```

Expected: `{"status":"ok"}`

- [ ] **Step 10: Commit**

```bash
git add api/Dockerfile api/requirements.txt api/main.py api/db.py api/tests/__init__.py api/tests/test_health.py docker-compose.yml
git commit -m "feat: add FastAPI skeleton with PostgreSQL model and health endpoint"
```

---

## Task 3: MinIO storage wrapper

**Files:**
- Create: `api/storage.py`
- Create: `api/tests/test_storage.py`

**Interfaces:**
- Produces:
  - `upload_bytes(key: str, data: bytes, content_type: str = "application/octet-stream") -> str` — uploads and returns the key
  - `presigned_url(key: str, expiry: int = 86400) -> str` — returns a presigned GET URL
  - `ensure_bucket() -> None` — idempotent bucket creation (called at startup)

- [ ] **Step 1: Write the failing test**

```python
# api/tests/test_storage.py
# Requires MinIO running at localhost:9000 with minioadmin/changeme
# Run: docker compose up storage -d

import os
os.environ["MINIO_ENDPOINT"] = "http://localhost:9000"
os.environ["MINIO_BUCKET"] = "test-furnitur-storage"
os.environ["MINIO_ROOT_USER"] = "minioadmin"
os.environ["MINIO_ROOT_PASSWORD"] = "changeme"

import httpx
import pytest


def test_upload_and_presign():
    from storage import ensure_bucket, upload_bytes, presigned_url

    ensure_bucket()
    key = upload_bytes("tests/hello.txt", b"hello world", "text/plain")
    assert key == "tests/hello.txt"

    url = presigned_url("tests/hello.txt")
    resp = httpx.get(url)
    assert resp.status_code == 200
    assert resp.content == b"hello world"


def test_upload_binary():
    from storage import ensure_bucket, upload_bytes, presigned_url

    ensure_bucket()
    data = bytes(range(256))
    upload_bytes("tests/binary.bin", data, "application/octet-stream")
    url = presigned_url("tests/binary.bin")
    resp = httpx.get(url)
    assert resp.content == data
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd api && pytest tests/test_storage.py -v
```

Expected: `ImportError: cannot import name 'ensure_bucket' from 'storage'`

- [ ] **Step 3: Write api/storage.py**

```python
# api/storage.py
from __future__ import annotations

import io
import os

import boto3
from botocore.client import Config

MINIO_ENDPOINT = os.environ["MINIO_ENDPOINT"]
MINIO_BUCKET = os.environ["MINIO_BUCKET"]
MINIO_ROOT_USER = os.environ["MINIO_ROOT_USER"]
MINIO_ROOT_PASSWORD = os.environ["MINIO_ROOT_PASSWORD"]

_client = None


def _get_client():
    global _client
    if _client is None:
        _client = boto3.client(
            "s3",
            endpoint_url=MINIO_ENDPOINT,
            aws_access_key_id=MINIO_ROOT_USER,
            aws_secret_access_key=MINIO_ROOT_PASSWORD,
            config=Config(signature_version="s3v4"),
            region_name="us-east-1",
        )
    return _client


def ensure_bucket() -> None:
    client = _get_client()
    try:
        client.head_bucket(Bucket=MINIO_BUCKET)
    except client.exceptions.ClientError:
        client.create_bucket(Bucket=MINIO_BUCKET)


def upload_bytes(key: str, data: bytes, content_type: str = "application/octet-stream") -> str:
    _get_client().upload_fileobj(
        io.BytesIO(data),
        MINIO_BUCKET,
        key,
        ExtraArgs={"ContentType": content_type},
    )
    return key


def presigned_url(key: str, expiry: int = 86400) -> str:
    return _get_client().generate_presigned_url(
        "get_object",
        Params={"Bucket": MINIO_BUCKET, "Key": key},
        ExpiresIn=expiry,
    )
```

Also add `ensure_bucket()` call to `main.py` lifespan:

```python
# api/main.py — update lifespan
from storage import ensure_bucket

@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()
    ensure_bucket()
    yield
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd api && pytest tests/test_storage.py -v
```

Expected: both tests `PASSED`. Requires `docker compose up storage -d` running.

- [ ] **Step 5: Commit**

```bash
git add api/storage.py api/tests/test_storage.py api/main.py
git commit -m "feat: add MinIO storage wrapper with upload and presigned URL support"
```

---

## Task 4: Compute abstraction layer (migration-friendly seam)

**Files:**
- Create: `api/compute/__init__.py`
- Create: `api/compute/modal_backend.py`

**Interfaces:**
- Produces: `ComputeBackend` Protocol with these async methods:
  - `scrape(page_url: str, job_id: str) -> dict`
  - `analyze(scrape_result: dict, job_id: str) -> dict`
  - `crop(job_id: str, intel_result: dict) -> dict`
  - `prepare_image(image_bytes: bytes, job_id: str) -> dict` — returns synthetic crop_result for uploaded images (skips S0/S1/S2)
  - `reconstruct(job_id: str, crop_result: dict, intel_result: dict, quality: str) -> dict`
  - `scale(job_id: str, reconstruct_result: dict, intel_result: dict) -> dict`
  - `texture(job_id: str, scale_result: dict, crop_result: dict, intel_result: dict) -> dict`
  - `render(job_id: str, texture_result: dict) -> dict`
  - `fetch_files(job_id: str, rels: list[str]) -> dict[str, bytes]`
- Produces: `ModalBackend` class implementing `ComputeBackend`

**Why this exists:** Replacing Modal with AWS Batch later means writing a new `AWSBatchBackend` class in `api/compute/aws_backend.py` that implements the same Protocol. The orchestrator never changes.

- [ ] **Step 1: Write api/compute/__init__.py (the Protocol)**

```python
# api/compute/__init__.py
from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class ComputeBackend(Protocol):
    """
    Abstraction over the GPU compute layer.
    Implement this Protocol to swap Modal for AWS Batch, GCP, etc.
    All methods are async; implementations must not block the event loop.
    """

    async def scrape(self, page_url: str, job_id: str) -> dict:
        """S0: Scrape product page. Returns ScrapeResult dict."""
        ...

    async def analyze(self, scrape_result: dict, job_id: str) -> dict:
        """S1: VLM page analysis. Returns IntelligenceResult dict."""
        ...

    async def crop(self, job_id: str, intel_result: dict) -> dict:
        """S2: GroundingDINO + SAM2 crop. Returns CropResult dict."""
        ...

    async def prepare_image(self, image_bytes: bytes, job_id: str) -> tuple[dict, dict]:
        """
        Image-upload alternative to S0+S1+S2.
        Returns (intel_result, crop_result) — synthetic dicts, no URLs.
        """
        ...

    async def reconstruct(
        self, job_id: str, crop_result: dict, intel_result: dict, quality: str
    ) -> dict:
        """S3: InstantMesh reconstruction. Returns ReconstructResult dict."""
        ...

    async def scale(
        self, job_id: str, reconstruct_result: dict, intel_result: dict
    ) -> dict:
        """S4: Dimension-aware scaling. Returns ScaleResult dict."""
        ...

    async def texture(
        self, job_id: str, scale_result: dict, crop_result: dict, intel_result: dict
    ) -> dict:
        """S5: Texture fusion. Returns TextureResult dict."""
        ...

    async def render(self, job_id: str, texture_result: dict) -> dict:
        """S6: Orthographic render. Returns RenderResult dict."""
        ...

    async def fetch_files(self, job_id: str, rels: list[str]) -> dict[str, bytes]:
        """Read artifact files from the compute backend's volume/storage."""
        ...
```

- [ ] **Step 2: Write api/compute/modal_backend.py**

```python
# api/compute/modal_backend.py
"""
Modal implementation of ComputeBackend.
Wraps all .remote() calls in run_in_executor to avoid blocking FastAPI's event loop.
"""
from __future__ import annotations

import asyncio
import uuid
from functools import partial

import modal

# Register all Modal stage functions with the app by importing them.
# The stages/ directory is copied into the Docker image by api/Dockerfile.
from stages.s0_scrape import scrape_page
from stages.s1_intelligence import analyze_page
from stages.s2_crop import Cropper
from stages.s3_reconstruct import InstantMeshGenerator
from stages.s4_scale import scale_mesh
from stages.s5_texture import TextureFuser
from stages.s6_render import render_views
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
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, fn, *args)


class ModalBackend:
    """Implements ComputeBackend using the Modal SDK."""

    async def scrape(self, page_url: str, job_id: str) -> dict:
        return await _run_in_executor(scrape_page.remote, page_url, job_id)

    async def analyze(self, scrape_result: dict, job_id: str) -> dict:
        return await _run_in_executor(analyze_page.remote, scrape_result, job_id)

    async def crop(self, job_id: str, intel_result: dict) -> dict:
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
        fn = partial(InstantMeshGenerator().generate.remote, job_id, crop_result, intel_result, quality=quality)
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, fn)

    async def scale(self, job_id: str, reconstruct_result: dict, intel_result: dict) -> dict:
        dims = intel_result.get("dimensions_mm")
        source = intel_result.get("dimensions_source", "absent")
        return await _run_in_executor(scale_mesh.remote, job_id, reconstruct_result, dims, source)

    async def texture(
        self, job_id: str, scale_result: dict, crop_result: dict, intel_result: dict
    ) -> dict:
        fn = partial(TextureFuser().fuse.remote, job_id, scale_result, crop_result, intel_result)
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, fn)

    async def render(self, job_id: str, texture_result: dict) -> dict:
        return await _run_in_executor(render_views.remote, job_id, texture_result)

    async def fetch_files(self, job_id: str, rels: list[str]) -> dict[str, bytes]:
        return await _run_in_executor(_fetch_volume_files.remote, job_id, [r for r in rels if r])
```

- [ ] **Step 3: Verify Protocol compliance (no test runner needed — just a type check)**

```bash
cd api && python -c "
from compute import ComputeBackend
from compute.modal_backend import ModalBackend
assert isinstance(ModalBackend(), ComputeBackend), 'ModalBackend must implement ComputeBackend'
print('Protocol check passed')
"
```

Expected: `Protocol check passed`

- [ ] **Step 4: Commit**

```bash
git add api/compute/__init__.py api/compute/modal_backend.py
git commit -m "feat: add ComputeBackend Protocol and ModalBackend for AWS-migration-friendly compute abstraction"
```

---

## Task 5: Job CRUD endpoints

**Files:**
- Modify: `api/main.py` (add POST /jobs, GET /jobs, GET /jobs/{id}, DELETE /jobs/{id})
- Create: `api/tests/conftest.py`
- Create: `api/tests/test_jobs.py`

**Interfaces:**
- Consumes: `Job` model from `db.py`, `upload_bytes` + `presigned_url` from `storage.py`, `ModalBackend` from `compute/modal_backend.py`
- Produces:
  - `POST /jobs` → `{"id": str, "status": "queued"}` (201)
  - `GET /jobs` → list of job summary dicts
  - `GET /jobs/{id}` → full job detail dict with presigned asset URLs
  - `DELETE /jobs/{id}` → `{"id": str, "status": "cancelled"}` (200)

- [ ] **Step 1: Write the failing tests**

```python
# api/tests/conftest.py
import os
os.environ["DATABASE_URL"] = "sqlite+aiosqlite:///:memory:"
os.environ["MINIO_ENDPOINT"] = "http://localhost:9000"
os.environ["MINIO_BUCKET"] = "test-furnitur"
os.environ["MINIO_ROOT_USER"] = "minioadmin"
os.environ["MINIO_ROOT_PASSWORD"] = "changeme"

import asyncio
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
    main.backend = mock_backend   # inject mock before app starts
    # Patch ensure_bucket so unit tests don't need a real MinIO instance running
    with patch("main.ensure_bucket"):
        async with AsyncClient(transport=ASGITransport(app=main.app), base_url="http://test") as c:
            yield c
```

```python
# api/tests/test_jobs.py
import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_create_job_url(client: AsyncClient):
    response = await client.post("/jobs", data={"url": "https://ikea.com/sofa"})
    assert response.status_code == 201
    body = response.json()
    assert "id" in body
    assert body["status"] == "queued"


@pytest.mark.asyncio
async def test_create_job_image(client: AsyncClient):
    response = await client.post(
        "/jobs",
        files={"image": ("sofa.png", b"\x89PNG\r\n\x1a\n" + b"\x00" * 100, "image/png")},
    )
    assert response.status_code == 201
    body = response.json()
    assert body["status"] == "queued"


@pytest.mark.asyncio
async def test_create_job_no_input(client: AsyncClient):
    response = await client.post("/jobs")
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_list_jobs_empty(client: AsyncClient):
    response = await client.get("/jobs")
    assert response.status_code == 200
    assert isinstance(response.json(), list)


@pytest.mark.asyncio
async def test_get_job_not_found(client: AsyncClient):
    response = await client.get("/jobs/nonexistent-id")
    assert response.status_code == 404


@pytest.mark.asyncio
async def test_get_job_after_create(client: AsyncClient):
    create = await client.post("/jobs", data={"url": "https://example.com/chair"})
    job_id = create.json()["id"]
    response = await client.get(f"/jobs/{job_id}")
    assert response.status_code == 200
    body = response.json()
    assert body["id"] == job_id
    assert body["status"] == "queued"
    assert body["input_type"] == "url"
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd api && pytest tests/test_jobs.py -v
```

Expected: multiple failures — the routes don't exist yet.

- [ ] **Step 3: Add job routes to api/main.py**

Replace the entire `api/main.py` with:

```python
# api/main.py
from __future__ import annotations

import asyncio
import json
import uuid
from contextlib import asynccontextmanager
from datetime import datetime

from fastapi import BackgroundTasks, Depends, File, Form, HTTPException, UploadFile
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from compute import ComputeBackend
from compute.modal_backend import ModalBackend
from db import Job, SessionLocal, get_session, init_db
from orchestrator import run_pipeline, subscribe_job, unsubscribe_job
from storage import ensure_bucket, presigned_url, upload_bytes

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
        update(Job).where(Job.id == job_id).values(status="cancelled", updated_at=datetime.utcnow())
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
```

- [ ] **Step 4: Create api/orchestrator.py stub** (needed so main.py can import it)

```python
# api/orchestrator.py — stub (full implementation in Task 6)
import asyncio
from collections import defaultdict

_subscribers: dict[str, list[asyncio.Queue]] = defaultdict(list)


def subscribe_job(job_id: str, queue: asyncio.Queue) -> None:
    _subscribers[job_id].append(queue)


def unsubscribe_job(job_id: str, queue: asyncio.Queue) -> None:
    if queue in _subscribers.get(job_id, []):
        _subscribers[job_id].remove(queue)


def _push(job_id: str, event_type: str, data: dict) -> None:
    for q in list(_subscribers.get(job_id, [])):
        q.put_nowait({"type": event_type, "data": data})


async def run_pipeline(job_id: str, backend) -> None:
    pass  # replaced in Task 6
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
cd api && pip install aiosqlite && pytest tests/test_jobs.py tests/test_health.py -v
```

Expected: all tests `PASSED`.

- [ ] **Step 6: Commit**

```bash
git add api/main.py api/orchestrator.py api/tests/conftest.py api/tests/test_jobs.py
git commit -m "feat: add job CRUD endpoints (POST/GET /jobs, GET /jobs/{id}, DELETE)"
```

---

## Task 6: Pipeline orchestrator + SSE stream

**Files:**
- Modify: `api/orchestrator.py` (replace stub with full implementation)
- Create: `api/tests/test_orchestrator.py`

**Interfaces:**
- Consumes: `ComputeBackend` Protocol from `compute/__init__.py`, `MockBackend` from `conftest.py`
- Consumes: `Job` model + `SessionLocal` from `db.py`, `upload_bytes` + `presigned_url` from `storage.py`
- Produces: `run_pipeline(job_id: str, backend: ComputeBackend) -> None` (async, runs full pipeline, updates DB, pushes SSE)

- [ ] **Step 1: Write the failing tests**

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd api && pytest tests/test_orchestrator.py -v
```

Expected: failures because `run_pipeline` is still a stub (does nothing).

- [ ] **Step 3: Replace api/orchestrator.py with full implementation**

```python
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
from datetime import datetime

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
    kwargs["updated_at"] = datetime.utcnow()
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
        raise


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
```

Also add the missing exports to `api/storage.py` so orchestrator can import the constants:

```python
# At the top of api/storage.py, after the os.environ reads — these are already there,
# just confirm they are module-level names (they are).
# MINIO_ENDPOINT, MINIO_BUCKET, MINIO_ROOT_USER, MINIO_ROOT_PASSWORD are module-level.
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd api && pytest tests/test_orchestrator.py -v
```

Expected: all three orchestrator tests `PASSED`.

- [ ] **Step 5: Run all tests**

```bash
cd api && pytest tests/ -v
```

Expected: all tests pass. (test_storage.py requires `docker compose up storage -d`)

- [ ] **Step 6: Commit**

```bash
git add api/orchestrator.py api/tests/test_orchestrator.py
git commit -m "feat: add pipeline orchestrator with SSE push and stage-by-stage MinIO uploads"
```

---

## Task 7: Full docker-compose integration + smoke test

**Files:**
- Modify: `docker-compose.yml` (confirm api service is complete, verify all environment wiring)

**Interfaces:**
- Produces: all four services running together; end-to-end job submission verified with `curl`

- [ ] **Step 1: Verify the complete docker-compose.yml is correct**

The final `docker-compose.yml` should look like:

```yaml
services:
  db:
    image: postgres:16-alpine
    environment:
      POSTGRES_DB: furnitur
      POSTGRES_USER: furnitur
      POSTGRES_PASSWORD: ${POSTGRES_PASSWORD}
    volumes:
      - pgdata:/var/lib/postgresql/data
    ports:
      - "5432:5432"
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U furnitur -d furnitur"]
      interval: 5s
      timeout: 5s
      retries: 10

  storage:
    image: minio/minio:latest
    ports:
      - "9000:9000"
      - "9001:9001"
    command: server /data --console-address ":9001"
    environment:
      MINIO_ROOT_USER: ${MINIO_ROOT_USER}
      MINIO_ROOT_PASSWORD: ${MINIO_ROOT_PASSWORD}
    volumes:
      - miniodata:/data
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:9000/minio/health/live"]
      interval: 5s
      timeout: 5s
      retries: 10

  api:
    build:
      context: .
      dockerfile: api/Dockerfile
    ports:
      - "8000:8000"
    env_file: .env
    environment:
      DATABASE_URL: postgresql+asyncpg://furnitur:${POSTGRES_PASSWORD}@db:5432/furnitur
    depends_on:
      db:
        condition: service_healthy
      storage:
        condition: service_healthy
    volumes:
      - ~/.modal:/root/.modal:ro

volumes:
  pgdata:
  miniodata:
```

- [ ] **Step 2: Build and start all services**

```bash
docker compose up --build -d
docker compose ps
```

Expected: `db`, `storage`, `api` all show `running` (or `healthy`).

- [ ] **Step 3: Health check**

```bash
curl -s http://localhost:8000/health
```

Expected: `{"status":"ok"}`

- [ ] **Step 4: Submit a URL job**

```bash
curl -s -X POST http://localhost:8000/jobs \
  -F "url=https://www.ikea.com/us/en/p/kivik-sofa-hillared-dark-blue-s89393642/" | jq .
```

Expected: `{"id": "...", "status": "queued"}`

- [ ] **Step 5: Watch SSE progress stream (in a separate terminal)**

```bash
JOB_ID="<id from step 4>"
curl -N "http://localhost:8000/jobs/${JOB_ID}/stream"
```

Expected: a stream of `stage_started` and `stage_completed` events as Modal runs each stage. This will take several minutes. Final event: `job_completed`.

- [ ] **Step 6: Verify job detail has all assets**

```bash
curl -s "http://localhost:8000/jobs/${JOB_ID}" | jq '{status, crop_urls, mesh_textured_glb_url, render_front_url}'
```

Expected: `status: "completed"`, all URL fields populated with presigned MinIO URLs.

- [ ] **Step 7: Verify a presigned crop URL opens**

```bash
CROP_URL=$(curl -s "http://localhost:8000/jobs/${JOB_ID}" | jq -r '.crop_urls[0]')
curl -I "$CROP_URL"
```

Expected: `HTTP/1.1 200 OK` with `Content-Type: image/png`.

- [ ] **Step 8: Submit an image upload job**

```bash
curl -s -X POST http://localhost:8000/jobs \
  -F "image=@grey_sofa.png;type=image/png" | jq .
```

Expected: `{"id": "...", "status": "queued"}`

- [ ] **Step 9: Commit**

```bash
git add docker-compose.yml
git commit -m "feat: complete backend stack — all services wired in docker-compose"
```

---

## Verification Checklist

Before declaring Plan 1 done:

- [ ] `pytest api/tests/ -v` — all tests pass (test_storage.py requires MinIO running)
- [ ] `docker compose ps` — all 3 services healthy
- [ ] `curl http://localhost:8000/health` → `{"status":"ok"}`
- [ ] URL job submission creates a row in PostgreSQL (`docker compose exec db psql -U furnitur -d furnitur -c "SELECT id, status FROM jobs;"`)
- [ ] SSE stream shows all 7 stages completing for a URL job
- [ ] Presigned MinIO URL for a crop image returns HTTP 200
- [ ] Image upload job completes (skips S0/S1/S2, runs S3–S6)
- [ ] MinIO console at `http://localhost:9001` shows `furnitur-pipeline` bucket with files under `jobs/{id}/`

---

## Notes for AWS Migration (future)

When migrating off Modal, implement `api/compute/aws_backend.py`:

```python
# api/compute/aws_backend.py (skeleton for future migration)
import boto3
from compute import ComputeBackend

class AWSBatchBackend:
    """
    Implement each method by submitting an AWS Batch job and polling for completion.
    Each stage runs in an ECR-hosted Docker image built from the GPU Dockerfiles
    (convert images/__init__.py Modal DSL → Dockerfiles).
    fetch_files() reads from S3 directly (stages write to S3 instead of Modal Volume).
    """

    async def scrape(self, page_url: str, job_id: str) -> dict:
        raise NotImplementedError

    # ... implement remaining methods
```

Then in `api/main.py`, change:
```python
# from:
backend: ComputeBackend = ModalBackend()
# to:
backend: ComputeBackend = AWSBatchBackend()
```

The orchestrator, DB, MinIO/S3, and frontend are untouched.
