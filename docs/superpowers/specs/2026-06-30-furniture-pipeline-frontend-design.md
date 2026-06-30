# Furniture Pipeline — Full-Stack Frontend Design

**Date:** 2026-06-30  
**Branch:** pipeline-e-multi-evidence  
**Status:** Approved

---

## Context

The existing pipeline (Pipeline E) converts a furniture product URL or image into a dimensioned, texture-fused 3D mesh via six GPU stages running on Modal. It is currently CLI-only (`modal run main.py`). This design adds a web frontend, persistent job history, and asset storage so that non-technical users can submit jobs and download results through a browser.

The key constraint: Modal cannot be self-hosted. GPU compute always runs on Modal's cloud. Everything else (frontend, API, database, object storage) runs in Docker and is fully self-hostable.

---

## System Architecture

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
│                    │ PostgreSQL │  │   MinIO     │    │ │
│                    │  port 5432 │  │  port 9000  │    │ │
│                    └────────────┘  └─────────────┘    │ │
└──────────────────────────────────────────────────────────┘
                              │ Modal Python SDK
                              ▼
                    ┌──────────────────────┐
                    │  Modal Cloud (GPU)   │
                    │  S0 → S1 → S2 → S3  │
                    │     → S4 → S5 → S6  │
                    │  Modal Volume        │
                    └──────────────────────┘
```

**Four Docker services:**
| Service | Image | Purpose |
|---------|-------|---------|
| `frontend` | Custom (Node 20) | Next.js 14 App Router SPA |
| `api` | Custom (Python 3.11) | FastAPI job runner + Modal orchestrator |
| `db` | `postgres:16-alpine` | Job metadata |
| `storage` | `minio/minio:latest` | Asset object store (S3-compatible) |

**Pipeline orchestration:** FastAPI calls each Modal stage function individually using `modal.Function.from_name()`. After each stage completes, FastAPI reads artifacts from the Modal Volume via `volume.read_file()`, uploads them to MinIO, writes S3 keys to PostgreSQL, and pushes an SSE event to connected browsers. **No changes to the existing stage code are required.**

---

## Project Layout

```
furnitur/
├── docker-compose.yml
├── .env.example
├── api/                        ← NEW: FastAPI service
│   ├── Dockerfile
│   ├── main.py                 ← FastAPI app + route definitions
│   ├── orchestrator.py         ← Modal stage calling + MinIO upload logic
│   ├── db.py                   ← SQLAlchemy async models + session
│   └── requirements.txt
├── frontend/                   ← NEW: Next.js service
│   ├── Dockerfile
│   ├── package.json
│   └── src/
│       ├── app/
│       │   ├── page.tsx                 ← Submit page (URL + image upload)
│       │   ├── jobs/page.tsx            ← Job history table
│       │   └── jobs/[id]/page.tsx       ← Results + SSE live updates
│       └── components/
│           ├── ModelViewer.tsx          ← @google/model-viewer wrapper
│           └── PipelineProgress.tsx     ← Stage progress indicator
├── docs/superpowers/specs/      ← this file
└── ... (existing: stages/, schemas/, images/, main.py, etc.)
```

---

## Data Layer

### PostgreSQL — `jobs` table

```sql
CREATE TABLE jobs (
  id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
  updated_at      TIMESTAMPTZ NOT NULL DEFAULT now(),

  -- Input
  input_type      TEXT NOT NULL CHECK (input_type IN ('url', 'image')),
  input_url       TEXT,        -- NULL for image uploads
  input_s3_key    TEXT,        -- NULL for URL jobs

  -- Lifecycle
  status          TEXT NOT NULL DEFAULT 'queued'
                  CHECK (status IN ('queued', 'running', 'completed', 'failed')),
  current_stage   SMALLINT NOT NULL DEFAULT 0,  -- 0–6
  error_message   TEXT,
  stage_timings   JSONB NOT NULL DEFAULT '{}',
  -- e.g. {"s2_crop": {"started_at": "...", "ended_at": "..."}}

  -- S1 intelligence metadata
  furniture_category   TEXT,
  dimensions_mm        JSONB,  -- {"width_mm": 900, "depth_mm": 450, "height_mm": 760}

  -- S2 outputs (parallel arrays)
  crop_s3_keys    TEXT[],
  mask_s3_keys    TEXT[],

  -- S3–S5 mesh outputs
  mesh_glb_s3_key          TEXT,
  mesh_scaled_glb_s3_key   TEXT,
  mesh_textured_glb_s3_key TEXT,
  uv_map_s3_key            TEXT,
  texture_atlas_s3_key     TEXT,

  -- S6 renders
  render_front_s3_key  TEXT,
  render_side_s3_key   TEXT,
  render_top_s3_key    TEXT,
  render_angled_s3_key TEXT,

  -- Modal tracking
  modal_call_id TEXT
);
```

### MinIO — bucket `furnitur-pipeline`

```
jobs/{job_id}/
  input.{ext}             ← uploaded image (input_type='image' only)
  crops/
    crop_0.png … crop_N.png    ← 512×512 RGBA (S2)
    mask_0.png … mask_N.png    ← grayscale alpha (S2)
  meshes/
    mesh.glb                   ← raw LRM output (S3)
    mesh_scaled.glb            ← dimension-scaled (S4)
    mesh_textured.glb          ← texture-fused (S5)
    uv_map.png                 ← 1024×1024 UV map (S3)
    texture_atlas.png          ← 2048×2048 texture atlas (S5)
  renders/
    front.png  side.png  top.png  angled.png   ← 1024×1024 (S6)
```

All files accessed via **presigned URLs** (24-hour expiry). No public bucket policy.

---

## API Contract

### Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/jobs` | Create job. JSON `{"url": "..."}` or `multipart/form-data` with `image` file. Returns `{"id": "...", "status": "queued"}`. |
| `GET` | `/jobs` | List all jobs newest-first. Returns array of job summaries. |
| `GET` | `/jobs/{id}` | Full job record with presigned MinIO URLs for all assets. |
| `GET` | `/jobs/{id}/stream` | SSE stream for live stage updates. |
| `DELETE` | `/jobs/{id}` | Mark job cancelled (Modal functions run to completion regardless). |

### SSE Event Protocol

```
event: stage_started
data: {"stage": "s2_crop", "stage_index": 2, "label": "Cropping images"}

event: stage_completed
data: {
  "stage": "s2_crop",
  "stage_index": 2,
  "duration_s": 18.4,
  "assets": {
    "crops": ["https://…/crop_0.png?X-Amz-…", "…"],
    "masks": ["…"]
  }
}

event: stage_completed
data: {
  "stage": "s6_render",
  "stage_index": 6,
  "assets": {
    "mesh_textured_glb": "https://…/mesh_textured.glb?…",
    "front": "…", "side": "…", "top": "…", "angled": "…"
  }
}

event: job_completed
data: {"job_id": "…", "total_duration_s": 187.2}

event: job_failed
data: {"stage": "s3_reconstruct", "error": "CUDA OOM"}
```

---

## Frontend Pages

### `/` — Submit
- Two-tab card: **Paste URL** (text input) / **Upload Image** (drag-and-drop, accepts PNG/JPG/AVIF/WEBP)
- Submit → `POST /jobs` → redirect to `/jobs/[id]`

### `/jobs` — History
- Table: first-crop thumbnail, furniture category, status badge, created time, link to detail
- Newest job first

### `/jobs/[id]` — Results
- **Progress bar:** 7 labeled steps (Scrape → Intelligence → Crop → Reconstruct → Scale → Texture → Render), each step lights up as SSE events arrive
- **Crops section** (appears on S2 `stage_completed`): grid of cropped PNGs, each with a download button
- **3D Viewer** (appears on S5 `stage_completed`): `<model-viewer>` component loading `mesh_textured.glb` presigned URL — orbitable, zoomable
- **Renders section** (appears on S6 `stage_completed`): 2×2 grid of front/side/top/angled PNGs
- **Downloads panel:** buttons for `mesh_textured.glb`, `texture_atlas.png`, `mesh_scaled.glb`
- **Metadata card:** furniture category + detected real-world dimensions

---

## Docker Compose

```yaml
services:
  frontend:
    build: ./frontend
    ports: ["3000:3000"]
    environment:
      NEXT_PUBLIC_API_URL: http://localhost:8000
    depends_on: [api]

  api:
    build: ./api
    ports: ["8000:8000"]
    env_file: .env
    depends_on: [db, storage]

  db:
    image: postgres:16-alpine
    environment:
      POSTGRES_DB: furnitur
      POSTGRES_USER: furnitur
      POSTGRES_PASSWORD: ${POSTGRES_PASSWORD}
    volumes: [pgdata:/var/lib/postgresql/data]

  storage:
    image: minio/minio:latest
    ports:
      - "9000:9000"   # S3 API
      - "9001:9001"   # MinIO web console
    command: server /data --console-address ":9001"
    environment:
      MINIO_ROOT_USER: ${MINIO_ROOT_USER}
      MINIO_ROOT_PASSWORD: ${MINIO_ROOT_PASSWORD}
    volumes: [miniodata:/data]

volumes:
  pgdata:
  miniodata:
```

### `.env.example`

```env
# PostgreSQL
POSTGRES_PASSWORD=changeme

# MinIO
MINIO_ROOT_USER=minioadmin
MINIO_ROOT_PASSWORD=changeme
MINIO_ENDPOINT=http://storage:9000
MINIO_BUCKET=furnitur-pipeline

# Modal (required — GPU pipeline is not self-hostable)
MODAL_TOKEN_ID=ak-...
MODAL_TOKEN_SECRET=as-...

# OpenRouter (used by S1 Intelligence stage)
OPENROUTER_API_KEY=sk-or-...
```

### Startup

```bash
cp .env.example .env     # fill in Modal + OpenRouter keys
docker compose up -d
# Frontend:     http://localhost:3000
# MinIO console: http://localhost:9001
```

---

## Self-Hostability Summary

| Component | Self-hostable? | Notes |
|-----------|---------------|-------|
| Next.js frontend | ✅ Yes | Docker container |
| FastAPI backend | ✅ Yes | Docker container |
| PostgreSQL | ✅ Yes | Docker container |
| MinIO storage | ✅ Yes | Docker container |
| Modal GPU pipeline | ⚠️ No | Requires Modal account + token. Modal does not offer self-hosted deployments. There is no Modal-compatible Docker image for self-hosting GPU compute. |

---

## Verification Plan

1. `docker compose ps` — all 4 containers healthy
2. MinIO console at `http://localhost:9001` — accessible, bucket `furnitur-pipeline` visible
3. `POST /jobs` with `grey_sofa.png` (already in repo) — assert `201` with `job_id`
4. `POST /jobs` with `{"url": "..."}` — assert `201`
5. `curl -N http://localhost:8000/jobs/{id}/stream` — observe `stage_started` / `stage_completed` events for each stage
6. After S2 `stage_completed` — MinIO has `jobs/{id}/crops/crop_*.png`; presigned URLs open in browser
7. After S5 — `/jobs/[id]` page renders `<model-viewer>` with textured GLB; orbit works
8. After S6 — 2×2 render grid visible
9. `/jobs` page — lists submitted jobs with status and thumbnail
10. Download buttons — each triggers a browser download of the correct file
