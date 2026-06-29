"""
S3 — InstantMesh Reconstruction
Runs the TencentARC/InstantMesh multi-view LRM to produce a textured GLB mesh.

Modes:
  • Multi-view (2–6 real images):
      Uses the crop images from S2 with camera parameters derived from
      the S1 view-label assignments.  The LRM sees real photographs from
      known angles — geometrically more reliable than hallucinated views.

  • Single-image fallback:
      If only one crop is available the full InstantMesh pipeline runs:
      Zero123++ first synthesises 5 additional views, then the LRM fuses all 6.
      This is still higher quality than TripoSR (Pipeline A) because
      InstantMesh's LRM was trained on a larger/more diverse dataset (Objaverse-XL).

GPU: A10G — instantmesh-large peaks at ~22 GB VRAM with 4 views at
512×512, which fits within the A10G's 24 GB. instantmesh-small (~12 GB) is
selected by passing quality="fast" to the Modal method.

Camera convention used here follows Zero123++ / InstantMesh:
  polar    = angle from zenith  (0°=top-down, 90°=equatorial)
  azimuth  = angle from front   (0°=front, 90°=right, 180°=back, 270°=left)
  radius   = 4.0 (canonical InstantMesh training distance)
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import modal
import numpy as np

from images import (
    ARTIFACTS_DIR,
    ARTIFACTS_VOLUME,
    INSTANTMESH_IMAGE,
    MODEL_DIR,
    app,
)
from schemas import CropResult, IntelligenceResult, ReconstructResult

# View label → (polar_deg, azimuth_deg)
VIEW_CAMERA_PARAMS: dict[str, tuple[float, float]] = {
    "front":              (90.0,   0.0),
    "back":               (90.0, 180.0),
    "side":               (90.0,  90.0),
    "angled_front_right": (75.0,  45.0),
    "angled_front_left":  (75.0, 315.0),
    "top":                ( 0.0,   0.0),
    "detail":             (75.0,  45.0),  # treat like angled if used in recon
    "lifestyle":          (75.0,  45.0),
    "unknown":            (90.0,   0.0),
}

INSTANTMESH_RADIUS = 4.0
INSTANTMESH_CKPT_LARGE = "TencentARC/InstantMesh"
INSTANTMESH_CKPT_SMALL = "TencentARC/InstantMesh"
INSTANTMESH_CONFIG_LARGE = "/opt/InstantMesh/configs/instant-mesh-large.yaml"
INSTANTMESH_CONFIG_SMALL = "/opt/InstantMesh/configs/instant-mesh-small.yaml"


def _polar_azimuth_to_c2w(
    polar_deg: float,
    azimuth_deg: float,
    radius: float = INSTANTMESH_RADIUS,
) -> np.ndarray:
    """Build a 4×4 camera-to-world matrix from spherical coordinates."""
    polar = math.radians(polar_deg)
    azimuth = math.radians(azimuth_deg)

    # Camera position (Z-up world, looking at origin)
    x = radius * math.sin(polar) * math.sin(azimuth)
    y = radius * math.sin(polar) * math.cos(azimuth)
    z = radius * math.cos(polar)
    eye = np.array([x, y, z], dtype=np.float32)

    target = np.zeros(3, dtype=np.float32)
    up = np.array([0.0, 0.0, 1.0], dtype=np.float32)

    forward = target - eye
    norm = np.linalg.norm(forward)
    if norm < 1e-8:
        # Edge case: top-down view — pick a stable up vector
        up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        forward = np.array([0.0, 0.0, -1.0], dtype=np.float32)
    else:
        forward /= norm

    right = np.cross(forward, up)
    right_norm = np.linalg.norm(right)
    if right_norm < 1e-8:
        right = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    else:
        right /= right_norm

    true_up = np.cross(right, forward)

    c2w = np.eye(4, dtype=np.float32)
    c2w[:3, 0] = right
    c2w[:3, 1] = true_up
    c2w[:3, 2] = -forward
    c2w[:3, 3] = eye
    return c2w


@app.cls(
    image=INSTANTMESH_IMAGE,
    gpu="A10G",
    volumes={"/vol": ARTIFACTS_VOLUME},
    timeout=900,
    scaledown_window=300,
)
class InstantMeshGenerator:
    @modal.enter()
    def setup(self):
        import torch
        from omegaconf import OmegaConf

        self.torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self._configs: dict[str, object] = {}
        self._models: dict[str, object] = {}

        for quality, cfg_path in [
            ("large", INSTANTMESH_CONFIG_LARGE),
            ("small", INSTANTMESH_CONFIG_SMALL),
        ]:
            try:
                config = OmegaConf.load(cfg_path)
                self._configs[quality] = config
            except Exception as exc:
                print(f"[S3] Could not load {quality} config: {exc}")

    def _get_model(self, quality: str = "large"):
        """Lazy-load the requested model variant."""
        if quality in self._models:
            return self._models[quality]

        import sys
        sys.path.insert(0, "/opt/InstantMesh")

        import torch
        from omegaconf import OmegaConf
        from src.utils.train_util import instantiate_from_config

        config = self._configs[quality]
        model = instantiate_from_config(config.model_config)

        # Download weights via huggingface_hub
        from huggingface_hub import hf_hub_download
        ckpt_repo = INSTANTMESH_CKPT_LARGE if quality == "large" else INSTANTMESH_CKPT_SMALL
        ckpt_name = (
            "instant_mesh_large.ckpt"
            if quality == "large"
            else "instant_mesh_small.ckpt"
        )
        model_root = MODEL_DIR / "instantmesh"
        model_root.mkdir(parents=True, exist_ok=True)
        ckpt_path = hf_hub_download(
            repo_id=ckpt_repo,
            filename=ckpt_name,
            local_dir=str(model_root),
            local_dir_use_symlinks=False,
        )

        state_dict = torch.load(ckpt_path, map_location="cpu")
        state_dict = state_dict.get("state_dict", state_dict)
        model.load_state_dict(state_dict, strict=True)
        model = model.to(self.device).eval()

        self._models[quality] = model
        return model

    def _get_pipeline(self, quality: str = "large"):
        """Load the Zero123++ diffusion pipeline used for single-image mode."""
        key = f"pipeline_{quality}"
        if key in self._models:
            return self._models[key]

        import torch
        from diffusers import DiffusionPipeline

        config = self._configs[quality]
        pipe = DiffusionPipeline.from_pretrained(
            config.infer_config.unet_path,
            custom_pipeline=config.infer_config.custom_pipeline,
            torch_dtype=torch.float16,
        ).to(self.device)

        self._models[key] = pipe
        return pipe

    def _prepare_image_tensor(self, image_paths: list[Path]) -> "torch.Tensor":
        """Load crops and normalise to (1, N, 3, 320, 320) in [-1, 1]."""
        import torch
        import torchvision.transforms.functional as TF
        from PIL import Image

        imgs = []
        for p in image_paths:
            img = Image.open(p).convert("RGBA")
            # Fill alpha with white background (InstantMesh convention)
            bg = Image.new("RGB", img.size, (255, 255, 255))
            bg.paste(img, mask=img.split()[3])
            t = TF.to_tensor(TF.resize(bg, [320, 320]))  # (3, 320, 320) [0,1]
            imgs.append(t * 2 - 1)  # → [-1, 1]
        return torch.stack(imgs, dim=0).unsqueeze(0).to(self.device)  # (1, N, 3, H, W)

    def _build_camera_tensor(
        self,
        view_labels: list[str],
        radius: float = INSTANTMESH_RADIUS,
    ) -> "torch.Tensor":
        """
        Build camera embedding tensor expected by InstantMesh's LRM.
        Format: (1, N, 16) — each row is the flattened 4×4 c2w matrix.
        """
        import torch

        matrices = []
        for label in view_labels:
            polar, azimuth = VIEW_CAMERA_PARAMS.get(label, (90.0, 0.0))
            c2w = _polar_azimuth_to_c2w(polar, azimuth, radius)
            matrices.append(torch.from_numpy(c2w).float().flatten())  # (16,)
        cameras = torch.stack(matrices, dim=0).unsqueeze(0).to(self.device)  # (1, N, 16)
        return cameras

    @modal.method()
    def generate(
        self,
        job_id: str,
        crop_result: CropResult,
        intelligence: IntelligenceResult,
        quality: str = "large",
        texture_resolution: int = 1024,
    ) -> ReconstructResult:
        """
        Generate a textured 3D mesh using InstantMesh.

        When ≥ 2 reconstruction-candidate crops are available the LRM
        receives the real images with camera parameters derived from the
        S1 view-label assignments.  When only 1 crop is available the
        full pipeline (Zero123++ → LRM) runs in single-image mode.
        """
        import sys
        sys.path.insert(0, "/opt/InstantMesh")

        import torch

        ARTIFACTS_VOLUME.reload()
        job_dir = ARTIFACTS_DIR / job_id

        # Identify which crops are reconstruction candidates
        recon_urls = set(intelligence["reconstruction_candidates"])
        recon_crops = [
            c for c in crop_result["crops"]
            if c["source_url"] in recon_urls
        ]

        if not recon_crops:
            # Fall back to all available crops
            recon_crops = crop_result["crops"]

        # Sort by index to preserve canonical ordering
        recon_crops = sorted(recon_crops, key=lambda c: c["index"])

        crop_paths = [ARTIFACTS_DIR / c["crop_rel"] for c in recon_crops]
        view_labels = [c["view_label"] for c in recon_crops]

        model = self._get_model(quality)

        if len(crop_paths) == 1:
            # Single-image mode: run Zero123++ to synthesise 5 additional views
            print("[S3] Single crop — running Zero123++ for view synthesis")
            images_tensor, cameras_tensor = self._run_zero123_plus(
                crop_paths[0], quality
            )
        else:
            # Multi-view mode: use real images with known camera parameters
            print(f"[S3] Multi-view mode with {len(crop_paths)} crops: {view_labels}")
            images_tensor = self._prepare_image_tensor(crop_paths)
            cameras_tensor = self._build_camera_tensor(view_labels)

        with torch.no_grad():
            planes = model.forward_planes(images_tensor, cameras_tensor)
            mesh_result = model.extract_mesh(
                planes,
                use_texture_map=True,
                texture_resolution=texture_resolution,
            )

        # mesh_result is a tuple: (vertices, faces, uvs, mesh_tex_idx, tex_map)
        # or a trimesh object depending on InstantMesh version
        glb_path, obj_path, uv_path = self._export_mesh(mesh_result, job_dir)

        ARTIFACTS_VOLUME.commit()

        return {
            "job_id": job_id,
            "glb_rel": str(glb_path.relative_to(ARTIFACTS_DIR)),
            "obj_rel": str(obj_path.relative_to(ARTIFACTS_DIR)),
            "uv_map_rel": str(uv_path.relative_to(ARTIFACTS_DIR)),
        }

    def _run_zero123_plus(
        self,
        single_crop_path: Path,
        quality: str,
    ) -> tuple["torch.Tensor", "torch.Tensor"]:
        """
        Run Zero123++ on one crop to generate 6 canonical views,
        then return image + camera tensors ready for the LRM.
        """
        import sys
        sys.path.insert(0, "/opt/InstantMesh")

        import torch
        from PIL import Image
        from src.utils.camera_util import get_zero123plus_input_cameras

        pipeline = self._get_pipeline(quality)
        config = self._configs[quality]

        source_img = Image.open(single_crop_path).convert("RGBA")
        bg = Image.new("RGB", source_img.size, (255, 255, 255))
        bg.paste(source_img, mask=source_img.split()[3])
        source_rgb = bg

        with torch.no_grad():
            output = pipeline(
                source_rgb,
                num_inference_steps=config.infer_config.get("diff_steps", 75),
                guidance_scale=config.infer_config.get("guidance_scale", 4.0),
            ).images[0]

        # output is a single image with 6 views tiled in a 2×3 grid (320×320 each)
        view_images = self._split_zero123_grid(output)

        images_tensor = self._prepare_image_tensor_from_pil(view_images)
        cameras_tensor = get_zero123plus_input_cameras(
            batch_size=1, radius=INSTANTMESH_RADIUS
        ).to(self.device)

        return images_tensor, cameras_tensor

    def _split_zero123_grid(self, grid_image: "Image.Image") -> list["Image.Image"]:
        """Split the 2×3 Zero123++ output grid into 6 individual views."""
        from PIL import Image
        w, h = grid_image.size
        cell_w = w // 3
        cell_h = h // 2
        views = []
        for row in range(2):
            for col in range(3):
                box = (col * cell_w, row * cell_h, (col + 1) * cell_w, (row + 1) * cell_h)
                views.append(grid_image.crop(box))
        return views

    def _prepare_image_tensor_from_pil(self, images: list["Image.Image"]) -> "torch.Tensor":
        import torch
        import torchvision.transforms.functional as TF

        imgs = []
        for img in images:
            if img.mode != "RGB":
                img = img.convert("RGB")
            t = TF.to_tensor(TF.resize(img, [320, 320]))
            imgs.append(t * 2 - 1)
        return torch.stack(imgs, dim=0).unsqueeze(0).to(self.device)

    def _export_mesh(
        self,
        mesh_result: object,
        job_dir: Path,
    ) -> tuple[Path, Path, Path]:
        """Export the InstantMesh result to GLB, OBJ, and texture PNG."""
        import trimesh
        from PIL import Image as PILImage

        glb_path = job_dir / "mesh.glb"
        obj_path = job_dir / "mesh.obj"
        uv_path = job_dir / "uv_map.png"

        if isinstance(mesh_result, trimesh.Trimesh):
            mesh = mesh_result
            mesh.export(glb_path)
            mesh.copy().apply_scale([-1, 1, 1]).export(obj_path)
            # No separate UV map for plain trimesh output
            PILImage.new("RGB", (1024, 1024), (128, 128, 128)).save(uv_path)
        elif isinstance(mesh_result, (list, tuple)) and len(mesh_result) >= 5:
            # (vertices, faces, uvs, mesh_tex_idx, tex_map) format
            vertices, faces, uvs, mesh_tex_idx, tex_map = mesh_result[:5]

            import numpy as np
            import torch

            def _to_np(t):
                if isinstance(t, torch.Tensor):
                    return t.detach().cpu().numpy()
                return np.array(t)

            v = _to_np(vertices)
            f = _to_np(faces)
            uv = _to_np(uvs)
            tex = _to_np(tex_map)  # (H, W, 3) in [0, 1]

            mesh = trimesh.Trimesh(
                vertices=v,
                faces=f,
                process=False,
            )
            mesh.export(glb_path)
            mesh.copy().apply_scale([-1, 1, 1]).export(obj_path)

            if tex.ndim == 3 and tex.shape[-1] in (3, 4):
                tex_uint8 = (tex * 255).clip(0, 255).astype(np.uint8)
                PILImage.fromarray(tex_uint8).save(uv_path)
            else:
                PILImage.new("RGB", (1024, 1024), (128, 128, 128)).save(uv_path)
        else:
            raise RuntimeError(f"Unexpected mesh_result type: {type(mesh_result)}")

        return glb_path, obj_path, uv_path
