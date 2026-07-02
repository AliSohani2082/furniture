"""
Pipeline E — Multi-Evidence 3D Reconstruction
modal run main.py --page-url "https://example.com/sofa"
modal run main.py::test_local_image --image-path grey_sofa.png
"""
from __future__ import annotations

import json
import uuid
from pathlib import Path

import modal
import requests

# Import Modal app + shared infrastructure
from images import (
    ARTIFACTS_DIR,
    ARTIFACTS_VOLUME,
    BASE_IMAGE,
    APP_NAME,
    app,
)

# Import all stage symbols so their @app.function / @app.cls decorators
# register on `app` at import time
from stages.s0_scrape import scrape_page
from stages.s1_intelligence import analyze_page
from stages.s2_crop import Cropper
from stages.s3_reconstruct import InstantMeshGenerator
from stages.s4_scale import scale_mesh
from stages.s5_texture import TextureFuser
from stages.s6_render import render_views

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    )
}


# ---------------------------------------------------------------------------
# Artifact fetching helper (must run inside a Modal container for volume access)
# ---------------------------------------------------------------------------

@app.function(
    image=BASE_IMAGE,
    volumes={"/vol": ARTIFACTS_VOLUME},
    timeout=120,
)
def fetch_artifacts(job_id: str, rels: list[str]) -> dict[str, bytes]:
    ARTIFACTS_VOLUME.reload()
    return {
        rel: (ARTIFACTS_DIR / rel).read_bytes()
        for rel in rels
        if rel and (ARTIFACTS_DIR / rel).exists()
    }


def _save_artifacts(job_id: str, rels: list[str], out_root: Path) -> None:
    rels = [r for r in rels if r]
    files = fetch_artifacts.remote(job_id, rels)
    for rel, data in files.items():
        local_path = out_root / Path(rel).name
        local_path.parent.mkdir(parents=True, exist_ok=True)
        local_path.write_bytes(data)


# ---------------------------------------------------------------------------
# Core pipeline orchestration
# ---------------------------------------------------------------------------

def run_pipeline(
    page_url: str,
    out_dir: str = "outputs",
    quality: str = "large",
) -> dict:
    """
    Full Pipeline E run:
      S0 scrape → S1 VLM intelligence → S2 multi-crop →
      S3 InstantMesh reconstruct → S4 dimension scale →
      S5 texture fusion → S6 4-view render

    Returns the manifest dict and writes all artifacts to out_dir/job_id/.
    """
    job_id = uuid.uuid4().hex[:12]
    out_root = Path(out_dir) / job_id
    out_root.mkdir(parents=True, exist_ok=True)

    print(f"[pipeline-e] job_id={job_id}  url={page_url}")

    # S0 — Scrape
    scrape = scrape_page.remote(page_url, job_id)
    print(f"[S0] found {len(scrape['candidate_image_urls'])} candidate images")

    # S1 — VLM Page Intelligence
    intelligence = analyze_page.remote(scrape, job_id)
    print(
        f"[S1] category={intelligence['furniture_category']}  "
        f"recon_candidates={len(intelligence['reconstruction_candidates'])}  "
        f"texture_candidates={len(intelligence['texture_candidates'])}"
    )

    # S2 — Multi-Crop
    crop_result = Cropper().crop_all.remote(job_id, intelligence)
    print(f"[S2] cropped {len(crop_result['crops'])} images")

    # S3 — InstantMesh Reconstruction  (A10G)
    reconstruct = InstantMeshGenerator().generate.remote(
        job_id, crop_result, intelligence, quality=quality
    )
    print(f"[S3] mesh: {reconstruct['glb_rel']}")

    # S4 — Dimension-Aware Scaling
    scale = scale_mesh.remote(
        job_id,
        reconstruct,
        intelligence.get("dimensions_mm"),
        intelligence.get("dimensions_source", "absent"),
    )
    print(f"[S4] scale_applied={scale['scale_applied']}  factor={scale['scale_factor']}")

    # S5 — Texture Fusion  (A10)
    texture = TextureFuser().fuse.remote(
        job_id, scale, crop_result, intelligence
    )
    print(f"[S5] texture atlas: {texture['texture_atlas_rel']}")

    # S6 — Render 4 views  (CPU)
    renders = render_views.remote(job_id, texture)
    print(f"[S6] renders: front={renders['front_rel']}")

    # Collect all artifact relative paths
    all_rels = [
        # Crops (all of them)
        *[c["crop_rel"] for c in crop_result["crops"]],
        *[c["mask_rel"] for c in crop_result["crops"] if c.get("mask_rel")],
        # Mesh variants
        reconstruct["glb_rel"],
        reconstruct["obj_rel"],
        reconstruct["uv_map_rel"],
        scale["scaled_glb_rel"],
        scale["scaled_obj_rel"],
        texture["textured_glb_rel"],
        texture["textured_obj_rel"],
        texture["texture_atlas_rel"],
        # Renders
        renders["front_rel"],
        renders["side_rel"],
        renders["top_rel"],
        renders["angled_rel"],
    ]

    _save_artifacts(job_id, all_rels, out_root)

    manifest = {
        "job_id": job_id,
        "pipeline": APP_NAME,
        "page_url": page_url,
        "quality": quality,
        "intelligence": {
            "furniture_category": intelligence["furniture_category"],
            "dimensions_mm": intelligence.get("dimensions_mm"),
            "dimensions_source": intelligence.get("dimensions_source"),
            "reconstruction_candidates": intelligence["reconstruction_candidates"],
        },
        "crop": crop_result,
        "reconstruct": reconstruct,
        "scale": scale,
        "texture": texture,
        "renders": renders,
    }

    (out_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, default=str), encoding="utf-8"
    )
    print(f"[pipeline-e] done → {out_root}")
    return manifest


# ---------------------------------------------------------------------------
# Entry Points
# ---------------------------------------------------------------------------

@app.local_entrypoint()
def main(
    page_url: str,
    out_dir: str = "outputs",
    quality: str = "large",
):
    """
    Production entry point.

    Example:
        modal run main.py --page-url "https://www.ikea.com/us/en/p/kivik-sofa-..."
    """
    result = run_pipeline(page_url, out_dir=out_dir, quality=quality)
    print(json.dumps(result, indent=2, default=str))


@app.function(
    image=BASE_IMAGE,
    volumes={"/vol": ARTIFACTS_VOLUME},
    timeout=60,
)
def _upload_test_image(data: bytes, jid: str, src: str):
    job_dir = ARTIFACTS_DIR / jid
    job_dir.mkdir(parents=True, exist_ok=True)
    (job_dir / "input_0.jpg").write_bytes(data)
    (job_dir / "source_url.txt").write_text(src, encoding="utf-8")
    ARTIFACTS_VOLUME.commit()


@app.function(
    image=BASE_IMAGE,
    volumes={"/vol": ARTIFACTS_VOLUME},
    timeout=60,
)
def _prepare_crop_from_bytes(data: bytes, jid: str) -> str:
    from PIL import Image
    from io import BytesIO
    job_dir = ARTIFACTS_DIR / jid
    job_dir.mkdir(parents=True, exist_ok=True)
    img = Image.open(BytesIO(data)).convert("RGB")
    dest = job_dir / "raw_input.png"
    img.save(dest)
    ARTIFACTS_VOLUME.commit()
    return str(dest.relative_to(ARTIFACTS_DIR))


@app.local_entrypoint()
def test_local_image(
    image_path: str,
    out_dir: str = "outputs",
    quality: str = "large",
):
    """
    Bypass scraping — run the reconstruction pipeline on a local image file.
    Useful for unit-testing S2–S6 without a product page URL.

    Example:
        modal run main.py::test_local_image --image-path grey_sofa.png
    """
    job_id = uuid.uuid4().hex[:12]
    out_root = Path(out_dir) / job_id
    out_root.mkdir(parents=True, exist_ok=True)

    image_bytes = Path(image_path).read_bytes()

    _upload_test_image.remote(image_bytes, job_id, f"local:{image_path}")

    # Synthesise a minimal intelligence result for a single local image
    intelligence = {
        "furniture_category": "furniture",
        "material_hints": [],
        "dimensions_mm": None,
        "dimensions_source": "absent",
        "view_classifications": [
            {
                "url": f"local:{image_path}",
                "view": "front",
                "confidence": 1.0,
                "is_product_isolated": True,
            }
        ],
        "reconstruction_candidates": [f"local:{image_path}"],
        "texture_candidates": [f"local:{image_path}"],
    }

    raw_rel = _prepare_crop_from_bytes.remote(image_bytes, job_id)

    # Use a simplified crop_result wrapping the raw image
    # (production would run Cropper here; for test we skip detection)
    crop_result = {
        "job_id": job_id,
        "crops": [
            {
                "index": 0,
                "source_url": f"local:{image_path}",
                "view_label": "front",
                "crop_rel": raw_rel,
                "mask_rel": None,
                "fallback": True,
            }
        ],
    }

    # S3 onwards
    reconstruct = InstantMeshGenerator().generate.remote(
        job_id, crop_result, intelligence, quality=quality
    )
    scale = scale_mesh.remote(job_id, reconstruct, None, "absent")
    texture = TextureFuser().fuse.remote(job_id, scale, crop_result, intelligence)
    renders = render_views.remote(job_id, texture)

    all_rels = [
        raw_rel,
        reconstruct["glb_rel"], reconstruct["obj_rel"], reconstruct["uv_map_rel"],
        scale["scaled_glb_rel"], scale["scaled_obj_rel"],
        texture["textured_glb_rel"], texture["textured_obj_rel"],
        texture["texture_atlas_rel"],
        renders["front_rel"], renders["side_rel"],
        renders["top_rel"], renders["angled_rel"],
    ]
    _save_artifacts(job_id, all_rels, out_root)

    manifest = {
        "job_id": job_id,
        "pipeline": APP_NAME,
        "image_path": image_path,
        "quality": quality,
        "reconstruct": reconstruct,
        "scale": scale,
        "texture": texture,
        "renders": renders,
    }
    (out_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, default=str), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, default=str))


@app.local_entrypoint()
def test_local_image_s2(
    image_path: str,
    out_dir: str = "outputs",
    quality: str = "large",
):
    """
    Like test_local_image but starts from S2 (Cropper) so background removal
    is actually performed before reconstruction.

    Example:
        modal run main.py::test_local_image_s2 --image-path grey_sofa.png
    """
    job_id = uuid.uuid4().hex[:12]
    out_root = Path(out_dir) / job_id
    out_root.mkdir(parents=True, exist_ok=True)

    image_bytes = Path(image_path).read_bytes()

    # S2 — Crop with background removal
    crop_result = Cropper().crop_from_bytes.remote(job_id, image_bytes, view_label="front")
    print(f"[S2] cropped {len(crop_result['crops'])} images  fallback={crop_result['crops'][0]['fallback']}")

    intelligence = {
        "furniture_category": "furniture",
        "material_hints": [],
        "dimensions_mm": None,
        "dimensions_source": "absent",
        "view_classifications": [
            {
                "url": "local:bytes",
                "view": "front",
                "confidence": 1.0,
                "is_product_isolated": False,
            }
        ],
        "reconstruction_candidates": ["local:bytes"],
        "texture_candidates": ["local:bytes"],
    }

    # S3 onwards
    reconstruct = InstantMeshGenerator().generate.remote(
        job_id, crop_result, intelligence, quality=quality
    )
    scale = scale_mesh.remote(job_id, reconstruct, None, "absent")
    texture = TextureFuser().fuse.remote(job_id, scale, crop_result, intelligence)
    renders = render_views.remote(job_id, texture)

    zero123_grid_rel = f"{job_id}/zero123_grid.png"
    all_rels = [
        *[c["crop_rel"] for c in crop_result["crops"]],
        *[c["mask_rel"] for c in crop_result["crops"] if c.get("mask_rel")],
        zero123_grid_rel,
        reconstruct["glb_rel"], reconstruct["obj_rel"], reconstruct["uv_map_rel"],
        scale["scaled_glb_rel"], scale["scaled_obj_rel"],
        texture["textured_glb_rel"], texture["textured_obj_rel"],
        texture["texture_atlas_rel"],
        renders["front_rel"], renders["side_rel"],
        renders["top_rel"], renders["angled_rel"],
    ]
    _save_artifacts(job_id, all_rels, out_root)

    manifest = {
        "job_id": job_id,
        "pipeline": APP_NAME,
        "image_path": image_path,
        "quality": quality,
        "crop": crop_result,
        "reconstruct": reconstruct,
        "scale": scale,
        "texture": texture,
        "renders": renders,
    }
    (out_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, default=str), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, default=str))
