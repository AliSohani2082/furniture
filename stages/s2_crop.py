"""
S2 — Multi-Crop
Runs GroundingDINO + SAM2 on each reconstruction_candidate and
texture_candidate URL selected by S1.  Produces one RGBA crop per image
at 512×512 (InstantMesh's expected input size).

Adapted from Pipeline A's single-image Cropper class; the core detection
and segmentation logic is identical.  The main changes:
  - Iterates over a list of images rather than a single input.jpg
  - Writes crop_{n}.png / mask_{n}.png instead of crop.png / mask.png.
  - Returns a list of SingleCropMeta dicts, one per processed image.
"""
from __future__ import annotations

import json
from io import BytesIO
from pathlib import Path

import modal
import numpy as np
import requests
from PIL import Image

from images import ARTIFACTS_DIR, ARTIFACTS_VOLUME, CROPPER_IMAGE, MODEL_DIR, app
from schemas import CropResult, IntelligenceResult, SingleCropMeta

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    )
}

FURNITURE_LABELS = [
    "chair", "sofa", "armchair", "couch", "stool", "bench",
    "table", "desk", "bookshelf", "cabinet", "dresser",
    "nightstand", "wardrobe", "ottoman",
]


def _download_image(url: str) -> Image.Image:
    resp = requests.get(url, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    return Image.open(BytesIO(resp.content)).convert("RGB")


@app.cls(
    image=CROPPER_IMAGE,
    gpu="A10G",
    volumes={"/vol": ARTIFACTS_VOLUME},
    timeout=900,
    scaledown_window=300,
)
class Cropper:
    @modal.enter()
    def setup(self):
        import torch
        from huggingface_hub import hf_hub_download
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        from transformers import (
            AutoModelForZeroShotObjectDetection,
            AutoProcessor,
        )

        self.torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # GroundingDINO must stay in float32 — its deformable attention builds
        # an internal sampling grid in fp32 that conflicts with fp16 inputs to
        # grid_sample (see Pipeline A comments for full explanation).
        model_id = "IDEA-Research/grounding-dino-tiny"
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.detector = AutoModelForZeroShotObjectDetection.from_pretrained(
            model_id, torch_dtype=torch.float32,
        ).to(self.device).eval()

        model_root = MODEL_DIR / "sam2"
        model_root.mkdir(parents=True, exist_ok=True)
        checkpoint = hf_hub_download(
            repo_id="facebook/sam2.1-hiera-large",
            filename="sam2.1_hiera_large.pt",
            local_dir=str(model_root),
            local_dir_use_symlinks=False,
        )
        self.sam = SAM2ImagePredictor(
            build_sam2("configs/sam2.1/sam2.1_hiera_l.yaml", checkpoint, device=self.device)
        )

        import rembg
        self.rembg = rembg
        self.rembg_session = rembg.new_session()

    # ------------------------------------------------------------------
    # Core helpers
    # ------------------------------------------------------------------

    def _fit_square_rgba(
        self,
        rgba: Image.Image,
        target_size: int = 512,
        foreground_ratio: float = 0.82,
    ) -> Image.Image:
        """Paste the RGBA foreground centred on a square transparent canvas."""
        rgba = rgba.convert("RGBA")
        w, h = rgba.size
        scale = min(
            (target_size * foreground_ratio) / max(w, 1),
            (target_size * foreground_ratio) / max(h, 1),
        )
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))
        resized = rgba.resize((new_w, new_h), Image.Resampling.LANCZOS)
        canvas = Image.new("RGBA", (target_size, target_size), (0, 0, 0, 0))
        left = (target_size - new_w) // 2
        top = (target_size - new_h) // 2
        canvas.paste(resized, (left, top), resized)
        return canvas

    def _segment_image(
        self, image: Image.Image, labels: list[str]
    ) -> tuple[Image.Image, Image.Image | None, bool]:
        """
        Returns (rgba_crop, mask_img_or_None, used_fallback).
        Tries GroundingDINO → SAM2 first; falls back to rembg.
        """
        prompt_list = [f"a {x}" for x in labels]
        inputs = self.processor(
            images=image,
            text=[prompt_list],
            return_tensors="pt",
        ).to(self.device)

        model_dtype = next(self.detector.parameters()).dtype
        inputs = {
            k: v.to(model_dtype) if v.is_floating_point() else v
            for k, v in inputs.items()
        }

        with self.torch.no_grad():
            outputs = self.detector(**inputs)

        results = self.processor.post_process_grounded_object_detection(
            outputs,
            inputs["input_ids"],
            threshold=0.35,
            text_threshold=0.25,
            target_sizes=[(image.height, image.width)],
        )[0]

        if len(results["boxes"]) == 0:
            rgba = self.rembg.remove(image, session=self.rembg_session)
            return self._fit_square_rgba(rgba), None, True

        boxes = results["boxes"].detach().cpu().numpy()
        scores = results["scores"].detach().cpu().numpy()
        areas = (
            np.maximum(boxes[:, 2] - boxes[:, 0], 1)
            * np.maximum(boxes[:, 3] - boxes[:, 1], 1)
        )
        best_idx = int(np.argmax(scores * np.sqrt(areas)))
        box = boxes[best_idx]

        rgb = np.array(image)
        self.sam.set_image(rgb)
        masks, mask_scores, _ = self.sam.predict(
            box=box[None, :], multimask_output=True
        )
        if isinstance(masks, np.ndarray) and masks.ndim == 4:
            masks = masks[:, 0, :, :]

        best_mask = (masks[int(np.argmax(mask_scores))] > 0.5).astype(np.uint8)

        x1, y1, x2, y2 = [int(max(0, round(v))) for v in box]
        pad_x = int((x2 - x1) * 0.15)
        pad_y = int((y2 - y1) * 0.15)
        x1 = max(0, x1 - pad_x)
        y1 = max(0, y1 - pad_y)
        x2 = min(image.width, x2 + pad_x)
        y2 = min(image.height, y2 + pad_y)

        crop_rgb = rgb[y1:y2, x1:x2]
        crop_mask = best_mask[y1:y2, x1:x2]
        alpha = (crop_mask * 255).astype(np.uint8)
        rgba_arr = np.dstack([crop_rgb, alpha])
        rgba = Image.fromarray(rgba_arr, mode="RGBA")

        mask_img = Image.fromarray(alpha)
        return self._fit_square_rgba(rgba), mask_img, False

    # ------------------------------------------------------------------
    # Modal method
    # ------------------------------------------------------------------

    @modal.method()
    def crop_all(
        self,
        job_id: str,
        intelligence: IntelligenceResult,
        labels: list[str] | None = None,
    ) -> CropResult:
        """
        Crop every reconstruction_candidate and texture_candidate.
        De-duplicates URLs so each image is only processed once.
        """
        ARTIFACTS_VOLUME.reload()
        job_dir = ARTIFACTS_DIR / job_id
        job_dir.mkdir(parents=True, exist_ok=True)

        all_labels = labels or FURNITURE_LABELS

        # Gather all URLs, preserving order with de-dup
        seen: set[str] = set()
        ordered: list[tuple[str, str]] = []  # (url, role)
        for url in intelligence["reconstruction_candidates"]:
            if url and url not in seen:
                seen.add(url)
                # Derive view label from classification results
                view = next(
                    (vc["view"] for vc in intelligence["view_classifications"] if vc["url"] == url),
                    "front",
                )
                ordered.append((url, view))
        for url in intelligence["texture_candidates"]:
            if url and url not in seen:
                seen.add(url)
                view = next(
                    (vc["view"] for vc in intelligence["view_classifications"] if vc["url"] == url),
                    "detail",
                )
                ordered.append((url, view))

        crops: list[SingleCropMeta] = []

        for idx, (url, view_label) in enumerate(ordered):
            try:
                image = _download_image(url)
            except Exception as exc:
                print(f"[S2] Skipping {url}: {exc}")
                continue

            try:
                rgba, mask_img, fallback = self._segment_image(image, all_labels)
            except Exception as exc:
                print(f"[S2] Segmentation failed for {url}: {exc}")
                continue

            crop_path = job_dir / f"crop_{idx}.png"
            rgba.save(crop_path)

            mask_rel: str | None = None
            if mask_img is not None:
                mask_path = job_dir / f"mask_{idx}.png"
                mask_img.save(mask_path)
                mask_rel = str(mask_path.relative_to(ARTIFACTS_DIR))

            (job_dir / f"crop_meta_{idx}.json").write_text(
                json.dumps({"url": url, "view": view_label, "fallback": fallback}),
                encoding="utf-8",
            )

            crops.append({
                "index": idx,
                "source_url": url,
                "view_label": view_label,
                "crop_rel": str(crop_path.relative_to(ARTIFACTS_DIR)),
                "mask_rel": mask_rel,
                "fallback": fallback,
            })

        ARTIFACTS_VOLUME.commit()
        return {"job_id": job_id, "crops": crops}

    @modal.method()
    def crop_from_bytes(
        self,
        job_id: str,
        image_data: bytes,
        view_label: str = "front",
        labels: list[str] | None = None,
    ) -> CropResult:
        """
        Run the full S2 pipeline (GroundingDINO → SAM2 → rembg fallback)
        on raw image bytes instead of a URL. Used by the test_local_image_s2
        entrypoint to avoid skipping background removal.
        """
        from io import BytesIO

        ARTIFACTS_VOLUME.reload()
        job_dir = ARTIFACTS_DIR / job_id
        job_dir.mkdir(parents=True, exist_ok=True)

        all_labels = labels or FURNITURE_LABELS

        image = Image.open(BytesIO(image_data)).convert("RGB")

        rgba, mask_img, fallback = self._segment_image(image, all_labels)
        print(f"[S2] crop_from_bytes: fallback={fallback}")

        crop_path = job_dir / "crop_0.png"
        rgba.save(crop_path)

        mask_rel: str | None = None
        if mask_img is not None:
            mask_path = job_dir / "mask_0.png"
            mask_img.save(mask_path)
            mask_rel = str(mask_path.relative_to(ARTIFACTS_DIR))

        ARTIFACTS_VOLUME.commit()
        return {
            "job_id": job_id,
            "crops": [
                {
                    "index": 0,
                    "source_url": "local:bytes",
                    "view_label": view_label,
                    "crop_rel": str(crop_path.relative_to(ARTIFACTS_DIR)),
                    "mask_rel": mask_rel,
                    "fallback": fallback,
                }
            ],
        }
