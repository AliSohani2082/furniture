# api/compute/__init__.py
from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class ComputeBackend(Protocol):
    """
    Abstraction over the GPU compute layer.
    Implement this Protocol to swap Modal for AWS Batch, GCP, etc.
    All methods are async; implementations must not block the event loop.
    """

    async def scrape(self, page_url: str, job_id: str) -> dict:
        """S0: Scrape product page. Returns ScrapeResult dict."""
        ...

    async def analyze(self, scrape_result: dict, job_id: str) -> dict:
        """S1: VLM page analysis. Returns IntelligenceResult dict."""
        ...

    async def crop(self, job_id: str, intel_result: dict) -> dict:
        """S2: GroundingDINO + SAM2 crop. Returns CropResult dict."""
        ...

    async def prepare_image(self, image_bytes: bytes, job_id: str) -> tuple[dict, dict]:
        """
        Image-upload alternative to S0+S1+S2.
        Returns (intel_result, crop_result) — synthetic dicts, no URLs.
        """
        ...

    async def reconstruct(
        self, job_id: str, crop_result: dict, intel_result: dict, quality: str
    ) -> dict:
        """S3: InstantMesh reconstruction. Returns ReconstructResult dict."""
        ...

    async def scale(
        self, job_id: str, reconstruct_result: dict, intel_result: dict
    ) -> dict:
        """S4: Dimension-aware scaling. Returns ScaleResult dict."""
        ...

    async def texture(
        self, job_id: str, scale_result: dict, crop_result: dict, intel_result: dict
    ) -> dict:
        """S5: Texture fusion. Returns TextureResult dict."""
        ...

    async def render(self, job_id: str, texture_result: dict) -> dict:
        """S6: Orthographic render. Returns RenderResult dict."""
        ...

    async def fetch_files(self, job_id: str, rels: list[str]) -> dict[str, bytes]:
        """Read artifact files from the compute backend's volume/storage."""
        ...
