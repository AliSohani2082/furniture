"""
S4 — Dimension-Aware Scaling
If S1 extracted real-world dimensions from the product page, this stage scales
the mesh so its bounding box matches those dimensions.  This is a no-op when
dimensions are absent (dimensions_source == "absent").

Runs on CPU in BASE_IMAGE — trimesh bounding-box math is fast (<2 s).
Makes Pipeline E the only pipeline whose output is directly usable in AR/VR
and room-planning tools without manual scale correction.
"""
from __future__ import annotations

import json
from pathlib import Path

import modal

from images import ARTIFACTS_DIR, ARTIFACTS_VOLUME, BASE_IMAGE, app
from schemas import DimensionsMM, ReconstructResult, ScaleResult


@app.function(
    image=BASE_IMAGE,
    volumes={"/vol": ARTIFACTS_VOLUME},
    timeout=120,
)
def scale_mesh(
    job_id: str,
    reconstruct: ReconstructResult,
    dimensions_mm: DimensionsMM | None,
    dimensions_source: str,
) -> ScaleResult:
    """
    Scale the reconstructed mesh to match real-world dimensions.
    The mesh is scaled uniformly so its largest axis equals the corresponding
    dimension (width → X, depth → Y, height → Z in trimesh convention).
    """
    import trimesh

    ARTIFACTS_VOLUME.reload()
    job_dir = ARTIFACTS_DIR / job_id

    glb_src = ARTIFACTS_DIR / reconstruct["glb_rel"]
    obj_src = ARTIFACTS_DIR / reconstruct["obj_rel"]

    scaled_glb = job_dir / "mesh_scaled.glb"
    scaled_obj = job_dir / "mesh_scaled.obj"

    if not dimensions_mm or dimensions_source == "absent":
        # No dimensions available — copy mesh unchanged (symlink-style alias)
        scaled_glb.write_bytes(glb_src.read_bytes())
        scaled_obj.write_bytes(obj_src.read_bytes())
        ARTIFACTS_VOLUME.commit()
        return {
            "job_id": job_id,
            "scaled_glb_rel": str(scaled_glb.relative_to(ARTIFACTS_DIR)),
            "scaled_obj_rel": str(scaled_obj.relative_to(ARTIFACTS_DIR)),
            "scale_applied": False,
            "scale_factor": None,
            "dimensions_mm": None,
        }

    loaded = trimesh.load(glb_src, force="scene")
    if isinstance(loaded, trimesh.Scene):
        meshes = list(loaded.geometry.values())
        mesh = trimesh.util.concatenate(meshes) if meshes else trimesh.Trimesh()
    else:
        mesh = loaded

    if mesh.is_empty:
        raise RuntimeError("Empty mesh in scale_mesh")

    extents = mesh.bounding_box.extents  # (width_x, depth_y, height_z) in mesh units

    # Map dimension keys to mesh axes
    dim_map = {
        "width": (dimensions_mm.get("width"), extents[0]),
        "depth": (dimensions_mm.get("depth"), extents[1]),
        "height": (dimensions_mm.get("height"), extents[2]),
    }

    # Compute per-axis scale factors and pick the most constrained one
    scale_factors = []
    for key, (real_mm, mesh_extent) in dim_map.items():
        if real_mm and mesh_extent > 1e-8:
            scale_factors.append(real_mm / mesh_extent)

    if not scale_factors:
        # Dimensions were listed but none matched usable axes
        scaled_glb.write_bytes(glb_src.read_bytes())
        scaled_obj.write_bytes(obj_src.read_bytes())
        ARTIFACTS_VOLUME.commit()
        return {
            "job_id": job_id,
            "scaled_glb_rel": str(scaled_glb.relative_to(ARTIFACTS_DIR)),
            "scaled_obj_rel": str(scaled_obj.relative_to(ARTIFACTS_DIR)),
            "scale_applied": False,
            "scale_factor": None,
            "dimensions_mm": dimensions_mm,
        }

    # Use median scale factor to resist outlier dimension entries
    import numpy as np
    scale_factor = float(np.median(scale_factors))

    mesh_scaled = mesh.copy()
    mesh_scaled.apply_scale(scale_factor)
    mesh_scaled.export(scaled_glb)

    obj_mesh = mesh_scaled.copy()
    obj_mesh.apply_scale([-1, 1, 1])
    obj_mesh.export(scaled_obj)

    scale_meta = {
        "scale_factor": scale_factor,
        "extents_before": extents.tolist(),
        "extents_after": (mesh_scaled.bounding_box.extents).tolist(),
        "dimensions_mm": dimensions_mm,
        "dimensions_source": dimensions_source,
    }
    (job_dir / "scale_meta.json").write_text(
        json.dumps(scale_meta, indent=2), encoding="utf-8"
    )

    ARTIFACTS_VOLUME.commit()

    return {
        "job_id": job_id,
        "scaled_glb_rel": str(scaled_glb.relative_to(ARTIFACTS_DIR)),
        "scaled_obj_rel": str(scaled_obj.relative_to(ARTIFACTS_DIR)),
        "scale_applied": True,
        "scale_factor": scale_factor,
        "dimensions_mm": dimensions_mm,
    }
