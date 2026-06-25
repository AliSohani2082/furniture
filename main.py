import json
import math
import os
import re
import uuid
from io import BytesIO
from pathlib import Path
from urllib.parse import urljoin, urlparse

import modal
import numpy as np
import requests
from bs4 import BeautifulSoup
from PIL import Image, ImageOps

# =========================
# Modal App / Config
# =========================

APP_NAME = "pipeline-a-cheapest-fastest"
app = modal.App(APP_NAME)

ARTIFACTS_VOLUME = modal.Volume.from_name(
    "pipeline-a-artifacts", create_if_missing=True
)
ARTIFACTS_DIR = Path("/vol/artifacts")
MODEL_DIR = Path("/vol/models")

FURNITURE_LABELS = [
    "chair",
    "sofa",
    "armchair",
    "couch",
    "stool",
    "bench",
    "table",
    "desk",
    "bookshelf",
    "cabinet",
    "dresser",
    "nightstand",
    "wardrobe",
    "ottoman",
]

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    )
}

# Shared base layer (CUDA, system deps, common pip packages)
BASE_IMAGE = (
    modal.Image.from_registry(
        "nvidia/cuda:12.1.1-devel-ubuntu22.04", add_python="3.11"
    )
    .apt_install(
        "git", "build-essential", "cmake", "ninja-build", "ffmpeg",
        "libgl1", "libglib2.0-0", "libsm6", "libxext6", "libegl1",
    )
    .pip_install(
        "setuptools>=68", "wheel", "requests", "beautifulsoup4", "lxml",
        "pillow", "pillow-avif-plugin", "numpy", "trimesh", "pyrender",
        "pygltflib", "opencv-python-headless", "rembg", "accelerate",
        "huggingface_hub", "playwright", "pybind11", "scikit-build-core",
        "einops", "omegaconf", "imageio", "onnxruntime",
    )
    .run_commands("playwright install --with-deps chromium")
    .pip_install(
        "torch", "torchvision",
        index_url="https://download.pytorch.org/whl/cu121",
    )
    .env({
        "TORCH_CUDA_ARCH_LIST": "8.6",
        "CC": "/usr/bin/gcc",
        "CXX": "/usr/bin/g++",
    })
)

# Cropper image: GroundingDINO + SAM2 — wants a modern transformers
CROPPER_IMAGE = (
    BASE_IMAGE
    .pip_install("transformers>=4.53.2")
    .run_commands(
        "python -m pip install --no-build-isolation "
        "git+https://github.com/facebookresearch/sam2.git"
    )
)

# TripoSR image: needs a transformers old enough to predate the ViTModel
# attention refactor (q_proj/k_proj/v_proj/o_proj), or the released
# stabilityai/TripoSR checkpoint won't load via load_state_dict().
TRIPO_IMAGE = (
    BASE_IMAGE
    .pip_install("transformers==4.46.3")
    .run_commands(
        "python -m pip install --no-build-isolation "
        "git+https://github.com/tatsy/torchmcubes.git"
    )
    .run_commands(
        "git clone https://github.com/VAST-AI-Research/TripoSR.git /opt/TripoSR"
    )
    .env({"PYTHONPATH": "/opt/TripoSR"})
)

# =========================
# Helpers
# =========================

def _normalize_url(candidate: str, base_url: str) -> str:
    candidate = candidate.strip()
    if not candidate:
        return ""
    if candidate.startswith("//"):
        return "https:" + candidate
    return urljoin(base_url, candidate)


def _extract_jsonld_images(soup: BeautifulSoup, base_url: str) -> list[str]:
    out: list[str] = []
    for script in soup.select('script[type="application/ld+json"]'):
        raw = script.string or script.get_text(strip=True)
        if not raw:
            continue
        try:
            data = json.loads(raw)
        except Exception:
            continue

        stack = [data]
        while stack:
            item = stack.pop()
            if isinstance(item, dict):
                for key in ("image", "thumbnailUrl", "contentUrl", "url", "og:image"):
                    value = item.get(key)
                    if isinstance(value, str):
                        out.append(_normalize_url(value, base_url))
                    elif isinstance(value, list):
                        out.extend(
                            _normalize_url(v, base_url)
                            for v in value
                            if isinstance(v, str)
                        )
                    elif isinstance(value, dict):
                        stack.append(value)
                for value in item.values():
                    if isinstance(value, (dict, list)):
                        stack.append(value)
            elif isinstance(item, list):
                stack.extend(item)
    return [u for u in out if u]


def _extract_html_images(html: str, base_url: str) -> list[str]:
    soup = BeautifulSoup(html, "lxml")
    urls: list[str] = []

    meta_selectors = [
        'meta[property="og:image"]',
        'meta[property="og:image:url"]',
        'meta[name="twitter:image"]',
        'meta[property="twitter:image"]',
    ]
    for selector in meta_selectors:
        for tag in soup.select(selector):
            value = tag.get("content")
            if value:
                urls.append(_normalize_url(value, base_url))

    urls.extend(_extract_jsonld_images(soup, base_url))

    for tag in soup.find_all("img"):
        for attr in ("data-src", "data-lazy-src", "src"):
            value = tag.get(attr)
            if value:
                urls.append(_normalize_url(value, base_url))
        srcset = tag.get("srcset")
        if srcset:
            parts = [p.strip().split(" ")[0] for p in srcset.split(",") if p.strip()]
            urls.extend(_normalize_url(p, base_url) for p in parts)

    seen = set()
    deduped = []
    for url in urls:
        if url and url not in seen:
            seen.add(url)
            deduped.append(url)
    return deduped


def _write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def _make_job_dir(job_id: str) -> Path:
    job_dir = ARTIFACTS_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    return job_dir


# =========================
# Modal Functions
# =========================

@app.function(image=BASE_IMAGE, timeout=300)
def render_page(url: str) -> str:
    from playwright.sync_api import sync_playwright

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True, args=["--disable-dev-shm-usage"])
        page = browser.new_page(
            viewport={"width": 1440, "height": 2200},
            device_scale_factor=1,
        )
        page.goto(url, wait_until="networkidle", timeout=60_000)
        page.mouse.wheel(0, 3000)
        page.wait_for_timeout(1200)
        html = page.content()
        browser.close()
        return html


def pick_best_image_url(page_url: str) -> str:
    response = requests.get(page_url, headers=HEADERS, timeout=30)
    response.raise_for_status()
    urls = _extract_html_images(response.text, page_url)
    if urls:
        return urls[0]

    html = render_page.remote(page_url)
    urls = _extract_html_images(html, page_url)
    if urls:
        return urls[0]

    raise RuntimeError(f"No usable image URL found for {page_url}")


@app.function(
    image=BASE_IMAGE,
    volumes={"/vol": ARTIFACTS_VOLUME},
    timeout=120,
)
def persist_input_image(image_bytes: bytes, job_id: str, source_url: str) -> dict:
    job_dir = _make_job_dir(job_id)
    input_path = job_dir / "input.jpg"

    _write_bytes(input_path, image_bytes)
    (job_dir / "source_url.txt").write_text(source_url, encoding="utf-8")

    ARTIFACTS_VOLUME.commit()

    return {
        "job_id": job_id,
        "input_rel": str(input_path.relative_to(ARTIFACTS_DIR)),
    }


# =========================
# Cropper (GroundingDINO + SAM2)
# =========================

@app.cls(
    image=CROPPER_IMAGE,
    gpu="A10",
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

        # NOTE: GroundingDINO's deformable attention builds its sampling
        # grid internally in fp32 and feeds it straight into
        # nn.functional.grid_sample. grid_sample does not support fp16
        # inputs on this path, so loading the detector in float16 crashes
        # deep inside the model ("expected scalar type Half but found
        # Float") regardless of what dtype you cast pixel_values to before
        # calling it -- the mismatch happens on a tensor built *inside* the
        # layer, which the caller has no way to intercept. Running the
        # detector in float32 sidesteps this entirely. GroundingDINO-tiny
        # is small and detection isn't the pipeline's bottleneck (TripoSR
        # reconstruction is), so the fp32 cost here is negligible.
        model_id = "IDEA-Research/grounding-dino-tiny"
        self.processor = AutoProcessor.from_pretrained(model_id)
        
        self.detector = AutoModelForZeroShotObjectDetection.from_pretrained(
            model_id,
            torch_dtype=torch.float32,
        ).to(self.device).eval()
        
        model_root = MODEL_DIR / "sam2"
        model_root.mkdir(parents=True, exist_ok=True)

        # Only the checkpoint comes from HF Hub. The config name is NOT a
        # filesystem path: build_sam2() resolves `cfg` through Hydra's own
        # internal search path (provider=main, path=pkg://sam2), which only
        # sees configs bundled inside the installed `sam2` package. Passing
        # the absolute HF download path for the yaml makes Hydra try to
        # resolve that string as a config *name* under pkg://sam2, which is
        # exactly the MissingConfigException you hit.
        checkpoint = hf_hub_download(
            repo_id="facebook/sam2.1-hiera-large",
            filename="sam2.1_hiera_large.pt",
            local_dir=str(model_root),
            local_dir_use_symlinks=False,
        )
        cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"

        self.sam = SAM2ImagePredictor(
            build_sam2(cfg, checkpoint, device=self.device)
        )

    def _fit_square_rgba(
        self,
        rgba: Image.Image,
        target_size: int = 1024,
        foreground_ratio: float = 0.82,
    ) -> Image.Image:
        rgba = rgba.convert("RGBA")
        w, h = rgba.size
        scale = min(
            (target_size * foreground_ratio) / max(w, 1),
            (target_size * foreground_ratio) / max(h, 1),
        )
        new_size = (max(1, int(round(w * scale))), max(1, int(round(h * scale))))
        resized = rgba.resize(new_size, Image.Resampling.LANCZOS)

        canvas = Image.new("RGBA", (target_size, target_size), (0, 0, 0, 0))
        left = (target_size - resized.size[0]) // 2
        top = (target_size - resized.size[1]) // 2
        canvas.paste(resized, (left, top), resized)
        return canvas

    @modal.method()
    def crop(self, job_id: str, labels: list[str] | None = None) -> dict:
        ARTIFACTS_VOLUME.reload()

        job_dir = ARTIFACTS_DIR / job_id
        input_path = job_dir / "input.jpg"
        if not input_path.exists():
            raise FileNotFoundError(input_path)

        image = Image.open(input_path).convert("RGB")
        prompt_list = [f"a {x}" for x in (labels or FURNITURE_LABELS)]
        inputs = self.processor(
            images=image,
            text=[prompt_list],
            return_tensors="pt",
        ).to(self.device)

        # Detector now always runs in float32, so this cast is a no-op
        # safety net rather than a load-bearing fix -- kept in case the
        # model dtype ever changes again. input_ids/attention_mask are
        # left untouched since they must stay integer/long tensors.
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
            rgba = Image.open(input_path).convert("RGBA")
            out = self._fit_square_rgba(rgba)
            crop_path = job_dir / "crop.png"
            out.save(crop_path)
            ARTIFACTS_VOLUME.commit()
            return {
                "job_id": job_id,
                "crop_rel": str(crop_path.relative_to(ARTIFACTS_DIR)),
                "fallback": True,
            }

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
            box=box[None, :],
            multimask_output=True,
        )

        if isinstance(masks, np.ndarray) and masks.ndim == 4:
            masks = masks[:, 0, :, :]

        best_mask = masks[int(np.argmax(mask_scores))]
        best_mask = (best_mask > 0.5).astype(np.uint8)

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

        out = self._fit_square_rgba(rgba)

        crop_path = job_dir / "crop.png"
        mask_path = job_dir / "mask.png"

        out.save(crop_path)
        Image.fromarray(alpha).save(mask_path)

        (job_dir / "crop_meta.json").write_text(
            json.dumps(
                {
                    "box": [float(v) for v in box.tolist()],
                    "scores": [float(v) for v in scores.tolist()],
                    "labels": list(results["text_labels"]),
                },
                indent=2,
            ),
            encoding="utf-8",
        )

        ARTIFACTS_VOLUME.commit()

        return {
            "job_id": job_id,
            "crop_rel": str(crop_path.relative_to(ARTIFACTS_DIR)),
            "mask_rel": str(mask_path.relative_to(ARTIFACTS_DIR)),
            "fallback": False,
        }

# =========================
# TripoSR Generator
# =========================

@app.cls(
    image=TRIPO_IMAGE,
    gpu="A10",
    volumes={"/vol": ARTIFACTS_VOLUME},
    timeout=900,
    scaledown_window=300,
)
class TripoGenerator:
    @modal.enter()
    def setup(self):
        import torch
        from tsr.system import TSR

        self.torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # --- DEBUG: confirm which image_tokenizer module is actually loaded ---
        import tsr.models.tokenizers.image as img_mod
        print("=== tsr image tokenizer module path ===")
        print(img_mod.__file__)
        print("=== file contents (first 3000 chars) ===")
        print(open(img_mod.__file__).read()[:3000])
        print("=== end debug ===")
        # --- end debug ---

        self.model = TSR.from_pretrained(
            "stabilityai/TripoSR",
            config_name="config.yaml",
            weight_name="model.ckpt",
        )

        self.model.renderer.set_chunk_size(131072)
        self.model.to(self.device)

    def _fill_background(self, image: Image.Image) -> Image.Image:
        arr = np.array(image).astype(np.float32) / 255.0

        if arr.shape[-1] == 4:
            rgb = arr[:, :, :3] * arr[:, :, 3:4] + (1.0 - arr[:, :, 3:4]) * 0.5
        else:
            rgb = arr[:, :, :3]

        return Image.fromarray((rgb * 255.0).astype(np.uint8))

    @modal.method()
    def generate(self, job_id: str, mc_resolution: int = 256) -> dict:
        ARTIFACTS_VOLUME.reload()

        job_dir = ARTIFACTS_DIR / job_id
        crop_path = job_dir / "crop.png"
        if not crop_path.exists():
            raise FileNotFoundError(crop_path)

        image = Image.open(crop_path).convert("RGBA")
        image = self._fill_background(image)

        scene_codes = self.model(image, device=self.device)
        mesh = self.model.extract_mesh(scene_codes, resolution=mc_resolution)[0]

        from tsr.utils import to_gradio_3d_orientation

        mesh = to_gradio_3d_orientation(mesh)

        glb_path = job_dir / "mesh.glb"
        obj_path = job_dir / "mesh.obj"

        mesh.export(glb_path)

        mesh_obj = mesh.copy()
        mesh_obj.apply_scale([-1, 1, 1])
        mesh_obj.export(obj_path)

        ARTIFACTS_VOLUME.commit()

        return {
            "job_id": job_id,
            "glb_rel": str(glb_path.relative_to(ARTIFACTS_DIR)),
            "obj_rel": str(obj_path.relative_to(ARTIFACTS_DIR)),
        }


# =========================
# Rendering Views
# =========================

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
def render_views(job_id: str, size: int = 1024) -> dict:
    import os
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

    import pyrender
    import trimesh

    ARTIFACTS_VOLUME.reload()

    job_dir = ARTIFACTS_DIR / job_id
    glb_path = job_dir / "mesh.glb"
    if not glb_path.exists():
        raise FileNotFoundError(glb_path)

    loaded = trimesh.load(glb_path, force="scene")

    if isinstance(loaded, trimesh.Scene):
        meshes = list(loaded.geometry.values())
        mesh = trimesh.util.concatenate(meshes) if meshes else trimesh.Trimesh()
    else:
        mesh = loaded

    if mesh.is_empty:
        raise RuntimeError("Empty mesh")

    mesh = mesh.copy()
    mesh.apply_translation(-mesh.centroid)

    extents = mesh.bounding_box.extents
    radius = float(np.max(extents) * 0.5) if np.max(extents) > 0 else 1.0

    pyr_mesh = pyrender.Mesh.from_trimesh(mesh, smooth=False)

    def render_one(name, eye, up):
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

    front = render_one("front", np.array([0, -2.5 * radius, 0]), np.array([0, 0, 1]))
    side = render_one("side", np.array([2.5 * radius, 0, 0]), np.array([0, 0, 1]))
    top = render_one("top", np.array([0, 0, 2.5 * radius]), np.array([0, 1, 0]))

    ARTIFACTS_VOLUME.commit()

    return {
        "job_id": job_id,
        "front_rel": front,
        "side_rel": side,
        "top_rel": top,
    }


# =========================
# Pipeline Orchestration
# =========================

def _download_from_volume(rel_path: str, local_path: Path) -> None:
    local_path.parent.mkdir(parents=True, exist_ok=True)
    with open(local_path, "wb") as f:
        for chunk in ARTIFACTS_VOLUME.read_file(rel_path):
            f.write(chunk)


def run_pipeline(page_url: str, out_dir: str = "outputs", mc_resolution: int = 256) -> dict:
    job_id = uuid.uuid4().hex[:12]
    out_root = Path(out_dir) / job_id
    out_root.mkdir(parents=True, exist_ok=True)

    image_url = pick_best_image_url(page_url)
    image_bytes = requests.get(image_url, headers=HEADERS, timeout=30).content

    persist_input_image.remote(image_bytes, job_id, image_url)
    crop_info = Cropper().crop.remote(job_id, FURNITURE_LABELS)
    mesh_info = TripoGenerator().generate.remote(job_id, mc_resolution)
    view_info = render_views.remote(job_id)

    manifest = {
        "job_id": job_id,
        "page_url": page_url,
        "image_url": image_url,
        "crop": crop_info,
        "mesh": mesh_info,
        "views": view_info,
    }

    for rel in [
        crop_info["crop_rel"],
        crop_info.get("mask_rel"),
        mesh_info["glb_rel"],
        mesh_info["obj_rel"],
        view_info["front_rel"],
        view_info["side_rel"],
        view_info["top_rel"],
    ]:
        if rel:
            _download_from_volume(rel, out_root / Path(rel).name)

    (out_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )

    return manifest


# =========================
# Entry Point
# =========================

@app.local_entrypoint()
def main(page_url: str, out_dir: str = "outputs", mc_resolution: int = 256):
    result = run_pipeline(page_url, out_dir=out_dir, mc_resolution=mc_resolution)
    print(json.dumps(result, indent=2))


# Test endpoint (bypass scrapping)

@app.local_entrypoint()
def test_local_image(image_path: str, out_dir: str = "outputs", mc_resolution: int = 256):
    job_id = uuid.uuid4().hex[:12]
    out_root = Path(out_dir) / job_id
    out_root.mkdir(parents=True, exist_ok=True)

    image_bytes = Path(image_path).read_bytes()
    persist_input_image.remote(image_bytes, job_id, f"local:{image_path}")

    crop_info = Cropper().crop.remote(job_id, FURNITURE_LABELS)
    mesh_info = TripoGenerator().generate.remote(job_id, mc_resolution)
    view_info = render_views.remote(job_id)

    manifest = {"job_id": job_id, "crop": crop_info, "mesh": mesh_info, "views": view_info}
    for rel in [crop_info["crop_rel"], crop_info.get("mask_rel"), mesh_info["glb_rel"],
                mesh_info["obj_rel"], view_info["front_rel"], view_info["side_rel"], view_info["top_rel"]]:
        if rel:
            _download_from_volume(rel, out_root / Path(rel).name)

    print(json.dumps(manifest, indent=2))