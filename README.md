# Furniture Pipeline E

Converts a furniture product page URL or an uploaded image into a dimensioned,
texture-fused 3D mesh and four orthographic reference renders (front, side, top,
angled). Runs as a web application backed by Modal GPU compute.

---

## Contents

- [What it does](#what-it-does)
- [Architecture](#architecture)
- [Pipeline stages](#pipeline-stages)
- [Tech stack](#tech-stack)
- [Prerequisites](#prerequisites)
- [Quickstart — web app](#quickstart--web-app)
- [Quickstart — CLI](#quickstart--cli)
- [Configuration](#configuration)
- [Project structure](#project-structure)
- [API reference](#api-reference)
- [Development](#development)
- [Contact](#contact)

---

## What it does

Given a furniture product URL or a raw image, the pipeline:

1. **Scrapes** the page and extracts product images (URL path only).
2. **Analyses** page text and images with a VLM to detect furniture category,
   real-world dimensions, and best reconstruction candidates.
3. **Crops** the furniture item from each image using GroundingDINO + SAM 2,
   removing backgrounds and producing RGBA masks.
4. **Reconstructs** a 3D mesh from the best crop with InstantMesh (multi-view
   diffusion → LRM).
5. **Scales** the mesh to real-world dimensions extracted in step 2.
6. **Fuses textures** using the full set of source crops, producing a UV-mapped
   GLB with a high-resolution texture atlas.
7. **Renders** four orthographic views (front, side, top, angled) from the
   textured mesh.

**Output per job:** `mesh_textured.glb` + `texture_atlas.png` + four `.png`
renders, all stored in MinIO with presigned download URLs.

---

## Architecture

```
┌──────────────────────────────────────────────────────────┐
│                    Docker Compose                         │
│                                                           │
│  ┌─────────────┐    HTTP/SSE    ┌──────────────────────┐ │
│  │  Next.js 14 │ ◄────────────► │  FastAPI (Python)    │ │
│  │  port 3000  │               │  port 8000            │ │
│  └─────────────┘               └──────────┬───────────┘ │
│                                           │              │
│                              ┌────────────┼────────────┐ │
│                    ┌─────────▼──┐  ┌──────▼──────┐    │ │
│                    │ PostgreSQL │  │    MinIO    │    │ │
│                    │  port 5432 │  │  port 9000  │    │ │
│                    └────────────┘  └─────────────┘    │ │
└──────────────────────────────────────────────────────────┘
                              │ Modal Python SDK
                              ▼
                    ┌──────────────────────┐
                    │   Modal Cloud (GPU)  │
                    │  S0 → S1 → S2 → S3  │
                    │     → S4 → S5 → S6  │
                    │   Modal Volume       │
                    └──────────────────────┘
```

**Four Docker services:**

| Service | Image | Purpose |
|---|---|---|
| `frontend` | Custom (Node 20) | Next.js 14 App Router — submit, history, results |
| `api` | Custom (Python 3.11) | FastAPI — job queue, Modal orchestration, SSE |
| `db` | `postgres:16-alpine` | Job metadata and status |
| `storage` | `minio/minio:latest` | Artifact object store (S3-compatible) |

**GPU compute** always runs on Modal's cloud. The FastAPI service calls each
stage function via `.remote()`, reads artifacts from the Modal Volume, uploads
them to MinIO, updates PostgreSQL, and pushes live progress to browsers over
SSE. No changes to stage code are needed to swap the compute backend — the
`ComputeBackend` Protocol in `api/compute/` is the migration seam.

---

## Pipeline stages

| Stage | File | Model / tool | Output |
|---|---|---|---|
| S0 Scrape | `stages/s0_scrape.py` | `httpx` + `BeautifulSoup` | Candidate image URLs |
| S1 Intelligence | `stages/s1_intelligence.py` | OpenRouter VLM | Category, dimensions, view labels |
| S2 Crop | `stages/s2_crop.py` | GroundingDINO + SAM 2 | RGBA crops + masks |
| S3 Reconstruct | `stages/s3_reconstruct.py` | InstantMesh (LRM) | `.glb` mesh + UV map |
| S4 Scale | `stages/s4_scale.py` | geometry transform | Dimension-scaled `.glb` |
| S5 Texture | `stages/s5_texture.py` | multi-crop UV fusion | Textured `.glb` + atlas |
| S6 Render | `stages/s6_render.py` | Blender / trimesh | 4 × 1024 px orthographic PNGs |

---

## Tech stack

**Backend**
- Python 3.11, FastAPI 0.115+, SQLAlchemy 2 (async + asyncpg)
- Modal SDK (GPU compute)
- boto3 / botocore (MinIO / S3)
- PostgreSQL 16, MinIO

**Frontend**
- Next.js 14 (App Router), React 18, TypeScript strict
- Tailwind CSS, shadcn/ui (Radix primitives)
- `@google/model-viewer` (interactive 3D)
- Native `EventSource` API for SSE

---

## Prerequisites

- Docker + Docker Compose v2
- A [Modal](https://modal.com) account with `modal setup` completed on the host
- An [OpenRouter](https://openrouter.ai) API key (used by S1)
- Python 3.11+ (CLI path only)

---

## Quickstart — web app

```bash
git clone <repo-url>
cd furnitur

# 1. Configure credentials
cp .env.example .env
#    Set POSTGRES_PASSWORD, MINIO_ROOT_USER, MINIO_ROOT_PASSWORD,
#    MINIO_ENDPOINT, MINIO_BUCKET, OPENROUTER_API_KEY

# 2. Expose Modal credentials to the api container
#    (modal setup writes ~/.modal — mounted read-only into the container)
modal setup   # skip if already done on this machine

# 3. Start all services
docker compose up -d

# 4. Open the browser
open http://localhost:3000
```

| URL | Purpose |
|---|---|
| `http://localhost:3000` | Frontend (submit, history, results) |
| `http://localhost:8000/docs` | FastAPI interactive docs |
| `http://localhost:9001` | MinIO console |

---

## Quickstart — CLI

Run the pipeline directly on Modal without the web stack:

```bash
python -m venv myenv && source myenv/bin/activate
pip install -r requirements.txt
modal setup          # one-time auth
cp .env.example .env && source .env   # or export OPENROUTER_API_KEY=...

# From a product URL
modal run main.py --url "https://example.com/furniture-product"

# From a local image
modal run main.py --image grey_sofa.png
```

Outputs are written to the Modal Volume at `/vol/<job-id>/` and logged to
stdout. Use `modal volume get furnitur-artifacts <job-id>/` to pull files
locally.

---

## Configuration

All configuration is supplied via environment variables. Copy `.env.example`
to `.env` and fill in the values below. **Never commit `.env`.**

| Variable | Required | Description |
|---|---|---|
| `OPENROUTER_API_KEY` | Yes | OpenRouter key for S1 VLM calls |
| `POSTGRES_PASSWORD` | Yes (web) | PostgreSQL password |
| `MINIO_ROOT_USER` | Yes (web) | MinIO root username |
| `MINIO_ROOT_PASSWORD` | Yes (web) | MinIO root password |
| `MINIO_ENDPOINT` | Yes (web) | MinIO endpoint (`http://storage:9000` inside Docker) |
| `MINIO_BUCKET` | Yes (web) | Bucket name for pipeline artifacts |
| `DATABASE_URL` | Auto (web) | Set by `docker-compose.yml` from `POSTGRES_PASSWORD` |
| `NEXT_PUBLIC_API_URL` | Auto (web) | Set to `http://localhost:8000` in `docker-compose.yml` |

Modal credentials (`~/.modal/`) are mounted read-only into the `api` container.
No Modal environment variables need to be set manually.

---

## Project structure

```
furnitur/
├── docker-compose.yml
├── .env.example
├── main.py                      ← CLI entrypoint (modal run main.py)
├── requirements.txt             ← CLI / Modal dependencies
│
├── stages/                      ← Pipeline stage functions (Modal)
│   ├── s0_scrape.py
│   ├── s1_intelligence.py
│   ├── s2_crop.py
│   ├── s3_reconstruct.py
│   ├── s4_scale.py
│   ├── s5_texture.py
│   └── s6_render.py
│
├── schemas/                     ← Pydantic result schemas shared by stages
├── images/                      ← Modal image / volume definitions
│
├── api/                         ← FastAPI web service
│   ├── Dockerfile
│   ├── requirements.txt
│   ├── main.py                  ← Route handlers, lifespan
│   ├── db.py                    ← SQLAlchemy Job model, async session
│   ├── storage.py               ← MinIO client wrapper
│   ├── orchestrator.py          ← run_pipeline(), SSE push
│   ├── compute/
│   │   ├── __init__.py          ← ComputeBackend Protocol
│   │   └── modal_backend.py     ← ModalBackend implementation
│   └── tests/
│       ├── conftest.py          ← MockBackend, DB/MinIO fixtures
│       ├── test_health.py
│       ├── test_jobs.py
│       └── test_orchestrator.py
│
├── frontend/                    ← Next.js web frontend
│   ├── Dockerfile
│   ├── package.json
│   └── src/
│       ├── app/
│       │   ├── page.tsx         ← Submit page (URL + image upload)
│       │   ├── jobs/page.tsx    ← Job history table
│       │   └── jobs/[id]/page.tsx ← Results page with live SSE
│       ├── components/
│       │   ├── ui/              ← shadcn/ui generated components
│       │   ├── SubmitForm.tsx
│       │   ├── PipelineProgress.tsx
│       │   ├── CropGallery.tsx
│       │   ├── ModelViewer.tsx
│       │   ├── RenderGallery.tsx
│       │   ├── DownloadPanel.tsx
│       │   └── JobStatusBadge.tsx
│       └── lib/
│           ├── api.ts           ← Typed fetch wrappers
│           └── types.ts         ← JobSummary, JobDetail, SSE event types
│
└── docs/
    └── superpowers/
        ├── specs/               ← Architecture decision records
        └── plans/               ← Task-by-task implementation plans
```

---

## API reference

### Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Liveness check → `{"status": "ok"}` |
| `POST` | `/jobs` | Create job. Form field `url` **or** file field `image`. Returns `{"id": "...", "status": "queued"}` (201). |
| `GET` | `/jobs` | List all jobs, newest first. |
| `GET` | `/jobs/{id}` | Full job record with presigned MinIO URLs for all assets. |
| `GET` | `/jobs/{id}/stream` | SSE stream for live stage progress. |
| `DELETE` | `/jobs/{id}` | Mark job `cancelled`. Modal functions run to completion regardless. |

### SSE event protocol

```
event: stage_started
data: {"stage": "s2_crop", "stage_index": 2, "label": "Cropping images"}

event: stage_completed
data: {
  "stage": "s2_crop",
  "stage_index": 2,
  "assets": {
    "crops": ["https://…/crop_0.png?X-Amz-…"],
    "masks": ["…"]
  }
}

event: stage_completed
data: {
  "stage": "s6_render",
  "stage_index": 6,
  "assets": {
    "front": "…", "side": "…", "top": "…", "angled": "…"
  }
}

event: job_completed
data: {"job_id": "…"}

event: job_failed
data: {"error": "CUDA OOM"}
```

---

## Development

### Backend tests

```bash
cd api
pip install -r requirements.txt aiosqlite
pytest tests/ -v
# test_storage.py requires: docker compose up storage -d
```

### Frontend type-check and dev server

```bash
cd frontend
cp .env.local.example .env.local
npm install
npx tsc --noEmit   # type check
npm run dev        # http://localhost:3000
```

### Adding a new compute backend

Implement the `ComputeBackend` Protocol in `api/compute/` and swap the
instantiation in `api/main.py`:

```python
# api/main.py
from compute.aws_backend import AWSBatchBackend
backend: ComputeBackend = AWSBatchBackend()
```

The orchestrator, database, MinIO layer, and frontend are untouched.

### Conventions

- Stage commits should reference the stage: `s3: reduce LRM chunk size to lower VRAM`
- Secrets are never committed. Use `.env.example` to document required variables.
- The `api/` Docker build context is the project root so `stages/`, `schemas/`,
  and `images/` can be copied into the container.

---

## Self-hostability

| Component | Self-hostable | Notes |
|---|---|---|
| Next.js frontend | Yes | Docker container |
| FastAPI backend | Yes | Docker container |
| PostgreSQL | Yes | Docker container |
| MinIO storage | Yes | Docker container |
| Modal GPU pipeline | No | Requires a Modal account. Modal does not offer self-hosted GPU deployments. |

---

## Contact

This is a private client engagement. Direct questions to the project maintainer.
