"""
S5 — Texture Fusion
Projects auxiliary product images onto the mesh's UV atlas using nvdiffrast
differentiable rasterization.  Fills in the hidden-surface texture gaps that
appear in all single-image reconstruction pipelines (A, B, D) — e.g. the back
and underside of a sofa that only has a front-facing hero shot.

Algorithm (UV-space projection baking):
  For each auxiliary image (texture_candidate not already used in reconstruction):
    1. Rasterize the mesh in UV space → every atlas texel knows its 3D world pos.
    2. Project those world positions to the auxiliary image's camera.
    3. Sample the auxiliary image at the projected positions (grid_sample).
    4. Blend: update each atlas texel only where the new image has a higher
       face-normal · camera-direction confidence score than the current value.
  Save the blended atlas as texture_atlas.png and re-export the mesh.

GPU: A10 (nvdiffrast is lightweight for furniture-scale meshes).
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import modal
import numpy as np

from images import ARTIFACTS_DIR, ARTIFACTS_VOLUME, TEXTURE_IMAGE, app
from schemas import CropResult, IntelligenceResult, ScaleResult, TextureResult
from stages.s3_reconstruct import VIEW_CAMERA_PARAMS, _polar_azimuth_to_c2w

TEXTURE_ATLAS_SIZE = 2048
INSTANTMESH_RADIUS = 4.0

# Pinhole intrinsics matching the canonical InstantMesh camera setup
# (35 mm focal length equivalent at 320×320 resolution)
_FX = _FY = 292.0  # ≈ 0.913 × 320
_CX = _CY = 160.0


@app.cls(
    image=TEXTURE_IMAGE,
    gpu="A10G",
    volumes={"/vol": ARTIFACTS_VOLUME},
    timeout=600,
    scaledown_window=180,
)
class TextureFuser:
    @modal.enter()
    def setup(self):
        import nvdiffrast.torch as dr
        import torch

        self.torch = torch
        self.dr = dr
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.glctx = dr.RasterizeCudaContext()

    # ------------------------------------------------------------------
    # Core baking helpers
    # ------------------------------------------------------------------

    def _load_mesh_geometry(
        self, glb_path: Path
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Returns (vertices, faces, uv_coords, uv_faces).
        Vertices are in world units (scaled by S4 if dimensions were available).
        """
        import trimesh

        loaded = trimesh.load(glb_path, force="scene")
        if isinstance(loaded, trimesh.Scene):
            meshes = list(loaded.geometry.values())
            mesh = trimesh.util.concatenate(meshes) if meshes else trimesh.Trimesh()
        else:
            mesh = loaded

        if mesh.is_empty:
            raise RuntimeError("Empty mesh in TextureFuser")

        vertices = np.array(mesh.vertices, dtype=np.float32)
        faces = np.array(mesh.faces, dtype=np.int32)

        # UV coordinates — use existing if present, otherwise generate planar
        if hasattr(mesh.visual, "uv") and mesh.visual.uv is not None:
            uv_coords = np.array(mesh.visual.uv, dtype=np.float32)
            uv_faces = faces  # same topology
        else:
            # Fallback: spherical UV unwrap
            norms = vertices - vertices.mean(axis=0)
            r = np.linalg.norm(norms, axis=1, keepdims=True) + 1e-8
            norms /= r
            u = 0.5 + np.arctan2(norms[:, 2], norms[:, 0]) / (2 * math.pi)
            v = 0.5 - np.arcsin(np.clip(norms[:, 1], -1, 1)) / math.pi
            uv_coords = np.stack([u, v], axis=-1).astype(np.float32)
            uv_faces = faces

        return vertices, faces, uv_coords, uv_faces

    def _bake_single_image(
        self,
        v_pos: "torch.Tensor",         # (N_v, 3) world-space vertices
        t_pos_idx: "torch.Tensor",      # (N_t, 3) geometry face indices
        uv_coords: "torch.Tensor",      # (N_uv, 2) UV coordinates [0,1]
        t_uv_idx: "torch.Tensor",       # (N_t, 3) UV face indices
        aux_image: "torch.Tensor",      # (3, H, W) auxiliary image [0,1]
        polar_deg: float,
        azimuth_deg: float,
        atlas_size: int = TEXTURE_ATLAS_SIZE,
    ) -> tuple["torch.Tensor", "torch.Tensor"]:
        """
        Project aux_image onto the mesh UV atlas.
        Returns (color_atlas, confidence_atlas) both in [atlas_size, atlas_size].
        """
        import torch
        import torch.nn.functional as F

        dr = self.dr
        device = self.device

        # Build camera matrices
        c2w = torch.from_numpy(
            _polar_azimuth_to_c2w(polar_deg, azimuth_deg, INSTANTMESH_RADIUS)
        ).to(device)  # (4, 4)
        w2c = torch.inverse(c2w)

        # ----------------------------------------------------------------
        # Step 1: UV-space rasterization
        # Place UV vertices in clip space (z=0, w=1) so nvdiffrast
        # rasterizes the UV atlas — each output texel maps to a triangle.
        # ----------------------------------------------------------------
        uv_clip = torch.cat([
            uv_coords * 2.0 - 1.0,                           # (N_uv, 2) → [-1,1]
            torch.zeros(uv_coords.shape[0], 1, device=device),  # z=0
            torch.ones(uv_coords.shape[0], 1, device=device),   # w=1
        ], dim=-1)  # (N_uv, 4)

        rast, _ = dr.rasterize(
            self.glctx, uv_clip[None], t_uv_idx, resolution=[atlas_size, atlas_size]
        )  # (1, H, W, 4)

        visible = (rast[0, ..., 3] > 0).float()  # (H, W) — which texels are covered

        # ----------------------------------------------------------------
        # Step 2: Interpolate world-space 3D positions at each atlas texel
        # ----------------------------------------------------------------
        world_pos, _ = dr.interpolate(
            v_pos[None], rast, t_pos_idx
        )  # (1, H, W, 3)
        world_pos = world_pos[0]  # (H, W, 3)

        # ----------------------------------------------------------------
        # Step 3: Project world positions to auxiliary image space
        # ----------------------------------------------------------------
        H_atlas, W_atlas = world_pos.shape[:2]
        ones = torch.ones(H_atlas, W_atlas, 1, device=device)
        h_pos = torch.cat([world_pos, ones], dim=-1)  # (H, W, 4)

        # World → camera
        cam_pos = (w2c @ h_pos.view(-1, 4).T).T.view(H_atlas, W_atlas, 4)
        cam_pos_xyz = cam_pos[..., :3]  # (H, W, 3)

        depth = cam_pos_xyz[..., 2]
        in_front = (depth > 0.01).float()  # mask: only points in front of camera

        # Pinhole projection
        px = (cam_pos_xyz[..., 0] / (depth + 1e-8)) * _FX + _CX
        py = (cam_pos_xyz[..., 1] / (depth + 1e-8)) * _FY + _CY

        H_img, W_img = aux_image.shape[1], aux_image.shape[2]
        grid_x = (px / W_img) * 2.0 - 1.0
        grid_y = (py / H_img) * 2.0 - 1.0
        # grid_sample convention: (x=W, y=H)
        grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0)  # (1, H, W, 2)

        # ----------------------------------------------------------------
        # Step 4: Sample auxiliary image colors at projected positions
        # ----------------------------------------------------------------
        color = F.grid_sample(
            aux_image.unsqueeze(0),  # (1, 3, H_img, W_img)
            grid,                    # (1, H_atlas, W_atlas, 2)
            mode="bilinear",
            padding_mode="border",
            align_corners=False,
        ).squeeze(0).permute(1, 2, 0)  # (H, W, 3)

        # ----------------------------------------------------------------
        # Step 5: Compute confidence = visible AND in_front
        # A simple proxy for viewing angle confidence.
        # ----------------------------------------------------------------
        confidence = visible * in_front  # (H, W)

        return color * confidence.unsqueeze(-1), confidence

    # ------------------------------------------------------------------
    # Modal method
    # ------------------------------------------------------------------

    @modal.method()
    def fuse(
        self,
        job_id: str,
        scale_result: ScaleResult,
        crop_result: CropResult,
        intelligence: IntelligenceResult,
        atlas_size: int = TEXTURE_ATLAS_SIZE,
    ) -> TextureResult:
        """
        For each texture_candidate not already used in reconstruction,
        project its crop onto the mesh UV atlas and blend with confidence.
        """
        import torch
        from PIL import Image as PILImage

        ARTIFACTS_VOLUME.reload()
        job_dir = ARTIFACTS_DIR / job_id

        glb_path = ARTIFACTS_DIR / scale_result["scaled_glb_rel"]

        # Load initial UV atlas from the mesh (baked by InstantMesh)
        uv_map_path = job_dir / "uv_map.png"
        if uv_map_path.exists():
            pil_tex = PILImage.open(uv_map_path).convert("RGB")
            if pil_tex.size != (atlas_size, atlas_size):
                pil_tex = pil_tex.resize((atlas_size, atlas_size), PILImage.LANCZOS)
            existing_tex = np.array(pil_tex, dtype=np.float32) / 255.0
        else:
            existing_tex = np.zeros((atlas_size, atlas_size, 3), dtype=np.float32)

        existing_conf = np.ones((atlas_size, atlas_size), dtype=np.float32) * 0.5

        # Load mesh geometry
        vertices, faces, uv_coords, uv_faces = self._load_mesh_geometry(glb_path)

        v_pos = torch.from_numpy(vertices).to(self.device)
        t_pos_idx = torch.from_numpy(faces).int().to(self.device)
        uv_t = torch.from_numpy(uv_coords).to(self.device)
        t_uv_idx = torch.from_numpy(uv_faces).int().to(self.device)

        # Identify texture candidates not already used in reconstruction
        recon_urls = set(intelligence["reconstruction_candidates"])
        texture_crops = [
            c for c in crop_result["crops"]
            if c["source_url"] not in recon_urls
               and c["view_label"] not in ("detail", "lifestyle", "unknown")
               or c["source_url"] in set(intelligence["texture_candidates"])
        ]

        if not texture_crops:
            # No extra images — export the existing UV map as-is
            print("[S5] No auxiliary texture candidates; using InstantMesh UV as-is")
        else:
            atlas_tensor = torch.from_numpy(existing_tex).to(self.device)
            conf_tensor = torch.from_numpy(existing_conf).to(self.device)

            for crop_meta in texture_crops:
                crop_path = ARTIFACTS_DIR / crop_meta["crop_rel"]
                try:
                    pil_img = PILImage.open(crop_path).convert("RGB")
                    import torchvision.transforms.functional as TF
                    aux_img = TF.to_tensor(pil_img).to(self.device)  # (3, H, W) [0,1]
                except Exception as exc:
                    print(f"[S5] Skipping {crop_path}: {exc}")
                    continue

                polar, azimuth = VIEW_CAMERA_PARAMS.get(
                    crop_meta["view_label"], (90.0, 0.0)
                )

                try:
                    new_color, new_conf = self._bake_single_image(
                        v_pos, t_pos_idx, uv_t, t_uv_idx,
                        aux_img, polar, azimuth, atlas_size,
                    )
                except Exception as exc:
                    print(f"[S5] Bake failed for {crop_meta['view_label']}: {exc}")
                    continue

                # Blend: update texels where new image has higher confidence
                update_mask = (new_conf > conf_tensor).float().unsqueeze(-1)
                atlas_tensor = atlas_tensor * (1 - update_mask) + new_color * update_mask
                conf_tensor = torch.maximum(conf_tensor, new_conf)

            existing_tex = atlas_tensor.cpu().numpy()

        # Save fused texture atlas
        tex_uint8 = (np.clip(existing_tex, 0, 1) * 255).astype(np.uint8)
        atlas_path = job_dir / "texture_atlas.png"
        PILImage.fromarray(tex_uint8).save(atlas_path)

        # Re-export mesh referencing the new atlas
        import trimesh

        loaded = trimesh.load(glb_path, force="scene")
        if isinstance(loaded, trimesh.Scene):
            meshes = list(loaded.geometry.values())
            mesh = trimesh.util.concatenate(meshes) if meshes else trimesh.Trimesh()
        else:
            mesh = loaded

        textured_glb = job_dir / "mesh_textured.glb"
        textured_obj = job_dir / "mesh_textured.obj"

        # Attach texture material
        material = trimesh.visual.material.SimpleMaterial(
            image=PILImage.fromarray(tex_uint8),
        )
        mesh.visual = trimesh.visual.TextureVisuals(
            uv=uv_coords if uv_coords is not None else None,
            material=material,
        )
        mesh.export(textured_glb)
        mesh.copy().apply_scale([-1, 1, 1]).export(textured_obj)

        ARTIFACTS_VOLUME.commit()

        return {
            "job_id": job_id,
            "textured_glb_rel": str(textured_glb.relative_to(ARTIFACTS_DIR)),
            "textured_obj_rel": str(textured_obj.relative_to(ARTIFACTS_DIR)),
            "texture_atlas_rel": str(atlas_path.relative_to(ARTIFACTS_DIR)),
        }
