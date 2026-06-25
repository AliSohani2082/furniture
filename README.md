# product-image-to-3d-pipeline

Serverless pipeline that converts a furniture product page URL into a 3D
mesh and three orthographic reference renders (front, side, top).

---

## What this does

Given a URL to a furniture product page (e.g. a retailer's product
listing), the pipeline:

1. **Scrapes** the page for product images.
2. **Detects and crops** the furniture item out of each image, removing
   background.
3. **Reconstructs a 3D mesh** of the item from the cropped image(s).
4. **Renders three orthographic views** (front, side, top) of the
   resulting mesh.

Output per job: one `.obj`/`.glb` mesh file, plus three `.png` renders.

This repository contains **multiple independent implementations** of that
pipeline, each making a different cost/latency/quality trade-off at the
3D-reconstruction stage. Each implementation lives on its own branch — see
[Branches](#branches) below. This file (the part you're reading now) is
identical across every branch; branch-specific architecture, setup, and
trade-off notes are appended below the `---` divider further down.

## Why multiple approaches

The 3D-reconstruction step is the one place in this pipeline where
cost, latency, and output quality genuinely trade off against each other,
and the right choice depends on the product's actual usage pattern
(volume, budget, how much visual fidelity the renders need). Rather than
guess, this repo implements several candidate architectures side by side
so they can be evaluated against real traffic before committing one to
production.

## Branches

| Branch | Approach | Summary |
|---|---|---|
| `pipeline-a-cheapest-fastest` | Mesh-first, single-image | Single-image reconstruction (TripoSR) → render views directly from the mesh. Lowest cost and latency; mesh/texture fidelity is the limiting factor. |
| `pipeline-b-balanced` | Mesh-first + refine | Higher-fidelity single-image reconstruction → render from mesh → diffusion img2img touch-up pass on each rendered view. |
| `pipeline-c-highest-quality` | Mesh-first, premium API | Premium closed-source 3D-generation API (highest fidelity, highest per-item cost) → native or rendered views. |
| `pipeline-d-views-first` | Views-first, multi-view reconstruction | Diffusion model generates the 3 novel views directly from the source photo first → all 4 images (original + 3 generated) are fused into one mesh by a multi-view reconstruction model. |

Each branch's README (below the divider) documents that branch's specific
architecture, tech stack, setup steps, environment variables, and known
limitations. **Do not merge these branches into one another** — they are
intentionally parallel, independent implementations sharing only this base
overview and, where noted, common infrastructure (object storage,
database). Promoting one approach to a long-lived `main`/production branch
is a deliberate later decision, not a side effect of merging.

## Shared infrastructure

All branches deploy onto the same three platforms and share the same job
schema, regardless of which 3D-reconstruction approach they implement:

- **Compute:** [Modal](https://modal.com) — serverless, scale-to-zero,
  Python-native. CPU-bound stages (scraping, detection) and GPU-bound
  stages (reconstruction, diffusion) run as separate container images so
  GPU cost is only paid for the stage(s) that actually need it.
- **Object storage:** Cloudflare R2 — S3-compatible, no egress fees.
  Stores scraped source images, crops, meshes, and rendered views.
- **Database:** [Neon](https://neon.tech) (hosted Postgres) — the `jobs`
  table doubles as both the job queue (via `SELECT ... FOR UPDATE SKIP
  LOCKED`) and the metadata/results store. No separate queue
  infrastructure is used at this project's traffic volume.

Connection details for all three are supplied via environment variables /
Modal Secrets — see the branch-specific README for the exact variable
names that branch's code expects.

## Repository conventions

- **Branches, not folders, separate approaches.** Each approach is a
  complete, independently deployable pipeline. This avoids one codebase
  accumulating conditional logic for four different architectures.
- **Commit messages** should state which pipeline stage they touch, e.g.
  `reconstruct: switch TripoSR chunk size to reduce VRAM`.
- **Secrets are never committed.** Use `.env.example` (committed) to
  document required variables, and an untracked `.env` / Modal Secret for
  actual values.
- **Each branch is deployed as its own Modal App** (distinct app name per
  branch) so they can run side by side on Modal without colliding or
  overwriting each other's deployments, volumes, or scheduled functions.

## Getting started

```bash
git clone <repo-url>
cd product-image-to-3d-pipeline
git checkout <branch-name>          # pick one of the branches listed above
pip install -r requirements.txt
modal setup                         # one-time Modal auth, if not already done
cp .env.example .env                # fill in R2 + Neon credentials
```

From here, follow the **branch-specific instructions below** for that
approach's exact deploy/run commands, since each branch has a different
Modal app entrypoint and may have additional setup steps (model downloads,
API keys for a third-party 3D-generation service, etc.).

## Contact

Questions about this repository should go to the maintainer directly
rather than through a public issue tracker, since this is a private
client engagement.

---