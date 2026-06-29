"""
S6 — Orthographic Render
Produces four reference renders of the textured mesh:
  front, side, top  — identical to Pipeline A's three views
  angled            — 3/4-angle view (azimuth 45°, elevation 20°), the most
                      photogenic perspective for marketing assets; not produced
                      by any other pipeline variant.

Operates on the texture-fused mesh from S5. Runs on CPU in BASE_IMAGE —
pyrender EGL offscreen rendering is fast (<10 s) and needs no GPU.
"""
from __future__ import annotations

from pathlib import Path

import modal
import numpy as np

from images import ARTIFACTS_DIR, ARTIFACTS_VOLUME, BASE_IMAGE, app
from schemas import RenderResult, TextureResult


def _look_at(eye: np.ndarray, target: np.ndarray, up: np.ndarray) -> np.ndarray:
    eye = np.asarray(eye, dtype=np.float32)
    target = np.asarray(target, dtype=np.float32)
    up = np.asarray(up, dtype=np.float32)

    forward = target - eye
    forward /= np.linalg.norm(forward) + 1e-8
    right = np.cross(forward, up)
    right /= np.linalg.norm(right) + 1e-8
    true_up = np.cross(right, forward)

    pose = np.eye(4, dtype=np.float32)
    pose[:3, 0] = right
    pose[:3, 1] = true_up
    pose[:3, 2] = -forward
    pose[:3, 3] = eye
    return pose


@app.function(
    image=BASE_IMAGE,
    volumes={"/vol": ARTIFACTS_VOLUME},
    timeout=300,
)
def render_views(
    job_id: str,
    texture_result: TextureResult,
    size: int = 1024,
) -> RenderResult:
    """Render four orthographic views of the textured mesh."""
    import os
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

    import pyrender
    import trimesh
    from PIL import Image

    ARTIFACTS_VOLUME.reload()

    job_dir = ARTIFACTS_DIR / job_id
    glb_path = ARTIFACTS_DIR / texture_result["textured_glb_rel"]

    if not glb_path.exists():
        raise FileNotFoundError(glb_path)

    loaded = trimesh.load(glb_path, force="scene")
    if isinstance(loaded, trimesh.Scene):
        meshes = list(loaded.geometry.values())
        mesh = trimesh.util.concatenate(meshes) if meshes else trimesh.Trimesh()
    else:
        mesh = loaded

    if mesh.is_empty:
        raise RuntimeError("Empty mesh in render_views")

    mesh = mesh.copy()
    mesh.apply_translation(-mesh.centroid)

    extents = mesh.bounding_box.extents
    radius = float(np.max(extents) * 0.5) if np.max(extents) > 0 else 1.0

    pyr_mesh = pyrender.Mesh.from_trimesh(mesh, smooth=False)

    def render_one(name: str, eye: np.ndarray, up: np.ndarray) -> str:
        scene = pyrender.Scene(
            bg_color=[1, 1, 1, 0],
            ambient_light=[0.35, 0.35, 0.35],
        )
        scene.add(pyr_mesh)

        camera = pyrender.OrthographicCamera(
            xmag=radius * 1.05,
            ymag=radius * 1.05,
        )
        scene.add(camera, pose=_look_at(eye, np.zeros(3), up))

        light = pyrender.DirectionalLight(color=np.ones(3), intensity=3.0)
        scene.add(light, pose=_look_at(
            np.array([2.5, 2.5, 2.5]),
            np.zeros(3),
            np.array([0.0, 0.0, 1.0]),
        ))

        renderer = pyrender.OffscreenRenderer(size, size)
        color, _ = renderer.render(scene)
        renderer.delete()

        out_path = job_dir / f"{name}.png"
        Image.fromarray(color).save(out_path)
        return str(out_path.relative_to(ARTIFACTS_DIR))

    r = 2.5 * radius

    front = render_one("front",  np.array([0.0, -r, 0.0]),             np.array([0, 0, 1]))
    side  = render_one("side",   np.array([r, 0.0, 0.0]),              np.array([0, 0, 1]))
    top   = render_one("top",    np.array([0.0, 0.0, r]),              np.array([0, 1, 0]))

    # Angled: 3/4 front-right, slight elevation
    import math
    az = math.radians(45)
    el = math.radians(20)
    angled_eye = np.array([
        r * math.cos(el) * math.sin(az),
        -r * math.cos(el) * math.cos(az),
        r * math.sin(el),
    ], dtype=np.float32)
    angled = render_one("angled", angled_eye, np.array([0, 0, 1]))

    ARTIFACTS_VOLUME.commit()

    return {
        "job_id": job_id,
        "front_rel": front,
        "side_rel": side,
        "top_rel": top,
        "angled_rel": angled,
    }
