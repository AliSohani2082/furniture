"""
Modal container image definitions and shared infrastructure for Pipeline E.
All stage modules import app, ARTIFACTS_VOLUME, ARTIFACTS_DIR, MODEL_DIR,
and their required image from here.
"""
from pathlib import Path

import modal

APP_NAME = "pipeline-e-multi-evidence"
app = modal.App(APP_NAME)

ARTIFACTS_VOLUME = modal.Volume.from_name(
    "pipeline-e-artifacts", create_if_missing=True
)
ARTIFACTS_DIR = Path("/vol/artifacts")
MODEL_DIR = Path("/vol/models")

# ---------------------------------------------------------------------------
# BASE_IMAGE — CPU stages: scraping, VLM call, scaling, rendering
# ---------------------------------------------------------------------------
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
        "openai",  # OpenRouter API call in S1
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
    .add_local_python_source("images", "schemas", "stages", copy=True)
)

# ---------------------------------------------------------------------------
# CROPPER_IMAGE — GPU A10G: GroundingDINO + SAM2 + CLIP fallback (S2)
# Identical to Pipeline A's CROPPER_IMAGE plus the CLIP model for
# image-classification fallback when HTML metadata is sparse.
# ---------------------------------------------------------------------------
CROPPER_IMAGE = (
    BASE_IMAGE
    .pip_install(
        "transformers>=4.53.2",
        "open-clip-torch",  # CLIP zero-shot image classification fallback
    )
    .run_commands(
        "SAM2_BUILD_CUDA=0 pip install --no-build-isolation "
        "git+https://github.com/facebookresearch/sam2.git"
    )
)

# ---------------------------------------------------------------------------
# INSTANTMESH_IMAGE — GPU A10G: multi-view LRM reconstruction (S3)
# xformers is CUDA-arch-specific; built for sm_86 (A10G). Pinned to
# transformers==4.40.0 as required by InstantMesh.
# ---------------------------------------------------------------------------
INSTANTMESH_IMAGE = (
    modal.Image.from_registry(
        "nvidia/cuda:12.1.1-devel-ubuntu22.04", add_python="3.11"
    )
    .apt_install(
        "git", "build-essential", "cmake", "ninja-build",
        "libgl1", "libglib2.0-0", "libsm6", "libxext6",
    )
    .pip_install(
        "setuptools>=68", "wheel", "numpy<2", "pillow", "einops",
        "omegaconf", "accelerate", "huggingface_hub", "trimesh",
        "pygltflib", "imageio", "scipy", "scikit-image", "xatlas",
        "opencv-python-headless", "pytorch-lightning==2.1.2",
        "PyMCubes", "plyfile", "torchmetrics", "rembg", "onnxruntime",
    )
    .pip_install(
        "torch==2.2.2", "torchvision==0.17.2",
        index_url="https://download.pytorch.org/whl/cu121",
    )
    .pip_install(
        "xformers==0.0.25.post1",
        index_url="https://download.pytorch.org/whl/cu121",
    )
    .pip_install("transformers==4.40.0", "diffusers==0.26.3", "huggingface_hub<0.24.0")
    .env({
        "TORCH_CUDA_ARCH_LIST": "8.6",  # sm_86 = A10G; must be set before nvdiffrast build
        "PYTHONPATH": "/opt/InstantMesh",
        "CC": "/usr/bin/gcc",
        "CXX": "/usr/bin/g++",
    })
    .run_commands(
        "git clone https://github.com/TencentARC/InstantMesh.git /opt/InstantMesh",
        # FlexiCubes is bundled inside InstantMesh at src/models/geometry/rep_3d/
        # — no separate install needed.
        "pip install --no-build-isolation git+https://github.com/NVlabs/nvdiffrast",
    )
    .add_local_python_source("images", "schemas", "stages")
)

# ---------------------------------------------------------------------------
# TEXTURE_IMAGE — GPU A10G: nvdiffrast UV-projection texture fusion (S5)
# nvdiffrast is not in Modal's internal PyPI mirror, so install from GitHub.
# ---------------------------------------------------------------------------
TEXTURE_IMAGE = (
    BASE_IMAGE
    .run_commands("pip install --no-build-isolation git+https://github.com/NVlabs/nvdiffrast")
)
