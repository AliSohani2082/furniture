from __future__ import annotations

from typing import Optional, TypedDict


class ScrapeResult(TypedDict):
    source_url: str
    page_text: str
    candidate_image_urls: list[str]


class ViewClassification(TypedDict):
    url: str
    view: str  # front|side|back|angled_front_right|angled_front_left|top|detail|lifestyle|unknown
    confidence: float
    is_product_isolated: bool


class DimensionsMM(TypedDict, total=False):
    width: float
    depth: float
    height: float


class IntelligenceResult(TypedDict):
    furniture_category: str
    material_hints: list[str]
    dimensions_mm: Optional[DimensionsMM]
    dimensions_source: str  # text_explicit | text_inferred | absent
    view_classifications: list[ViewClassification]
    reconstruction_candidates: list[str]
    texture_candidates: list[str]


class SingleCropMeta(TypedDict):
    index: int
    source_url: str
    view_label: str
    crop_rel: str
    mask_rel: Optional[str]
    fallback: bool


class CropResult(TypedDict):
    job_id: str
    crops: list[SingleCropMeta]


class ReconstructResult(TypedDict):
    job_id: str
    glb_rel: str
    obj_rel: str
    uv_map_rel: str


class ScaleResult(TypedDict):
    job_id: str
    scaled_glb_rel: str
    scaled_obj_rel: str
    scale_applied: bool
    scale_factor: Optional[float]
    dimensions_mm: Optional[DimensionsMM]


class TextureResult(TypedDict):
    job_id: str
    textured_glb_rel: str
    textured_obj_rel: str
    texture_atlas_rel: str


class RenderResult(TypedDict):
    job_id: str
    front_rel: str
    side_rel: str
    top_rel: str
    angled_rel: str
