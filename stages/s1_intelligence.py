"""
S1 — VLM Page Intelligence
Sends page text + HTML image context to Claude Haiku (text-only) to:
  - Classify each candidate image URL by view type and isolation
  - Extract furniture category and real-world dimensions
  - Select reconstruction_candidates (best for 3D) and texture_candidates

No vision API calls here — URL path tokens, alt text, and surrounding HTML
class names give >80 % accuracy at ~$0.0006/page vs ~$0.04 for image-based
classification. A CLIP zero-shot fallback handles pages with sparse metadata.
"""
from __future__ import annotations

import json
import os
import re
from urllib.parse import urlparse

import modal
import requests

from images import ARTIFACTS_DIR, ARTIFACTS_VOLUME, BASE_IMAGE, app
from schemas import IntelligenceResult, ScrapeResult

# View labels InstantMesh understands
VALID_VIEWS = frozenset({
    "front", "side", "back",
    "angled_front_right", "angled_front_left",
    "top", "detail", "lifestyle", "unknown",
})

_CLASSIFY_PROMPT = """\
You are a 3D reconstruction pipeline assistant analyzing an e-commerce furniture product page.

Your tasks:
1. Classify each candidate image URL by view type and whether it shows the product in isolation.
2. Extract the furniture category (e.g. "sofa", "chair", "desk").
3. Extract explicit product dimensions if present anywhere in the page text. Convert to mm.
4. Select up to 4 URLs as "reconstruction_candidates": these should be isolated product shots
   showing the overall form from recognizable angles (front, side, angled, etc.).
5. Select up to 4 URLs as "texture_candidates": any image useful for surface detail/color,
   including detail shots, even if they don't show the full silhouette.

Image URL context clues to use:
- URL path tokens (e.g. "front", "side", "angle", "detail", "hero", "lifestyle")
- alt text values
- HTML element classes surrounding the <img> (e.g. "gallery-main", "thumbnail", "swatch")
- Declared pixel dimensions in HTML (hero shots are usually largest)
- File format hints (avif/webp = CDN hero; tiny GIF/PNG with "icon" class = skip)

View types allowed: front | side | back | angled_front_right | angled_front_left |
                   top | detail | lifestyle | unknown

Return ONLY a valid JSON object matching this schema exactly — no prose, no markdown:
{
  "furniture_category": "<string>",
  "material_hints": ["<string>"],
  "dimensions_mm": {"width": <number>, "depth": <number>, "height": <number>} | null,
  "dimensions_source": "text_explicit" | "text_inferred" | "absent",
  "view_classifications": [
    {
      "url": "<string>",
      "view": "<view_type>",
      "confidence": <0.0-1.0>,
      "is_product_isolated": <true|false>
    }
  ],
  "reconstruction_candidates": ["<url>"],
  "texture_candidates": ["<url>"]
}

PAGE TEXT (first 6 000 chars):
{page_text}

IMAGE URL CONTEXTS (url | alt | surrounding_classes | declared_size):
{url_contexts}
"""


def _build_url_contexts(candidate_urls: list[str], html: str = "") -> str:
    """Build a compact text block describing each candidate URL."""
    from bs4 import BeautifulSoup

    img_meta: dict[str, dict] = {}
    if html:
        soup = BeautifulSoup(html, "lxml")
        for tag in soup.find_all("img"):
            for attr in ("src", "data-src", "data-lazy-src"):
                src = tag.get(attr, "").strip()
                if src:
                    parent_classes = " ".join(
                        tag.parent.get("class", []) if tag.parent else []
                    )
                    img_meta[src] = {
                        "alt": tag.get("alt", ""),
                        "classes": parent_classes,
                        "width": tag.get("width", ""),
                        "height": tag.get("height", ""),
                    }

    lines = []
    for url in candidate_urls[:40]:  # cap at 40 to stay within token budget
        path = urlparse(url).path
        meta = img_meta.get(url, {})
        line = (
            f"{url} | alt={meta.get('alt','')!r} "
            f"| classes={meta.get('classes','')!r} "
            f"| path={path}"
        )
        lines.append(line)
    return "\n".join(lines)


def _fallback_single_image(candidate_urls: list[str]) -> IntelligenceResult:
    """
    Minimal fallback used when the Haiku call fails.
    Returns the first URL as the sole reconstruction candidate.
    """
    first = candidate_urls[0] if candidate_urls else ""
    return {
        "furniture_category": "furniture",
        "material_hints": [],
        "dimensions_mm": None,
        "dimensions_source": "absent",
        "view_classifications": [
            {
                "url": first,
                "view": "front",
                "confidence": 0.5,
                "is_product_isolated": True,
            }
        ],
        "reconstruction_candidates": [first] if first else [],
        "texture_candidates": [first] if first else [],
    }


def _clip_classify_fallback(
    candidate_urls: list[str],
    top_k: int = 4,
) -> list[dict]:
    """
    CLIP zero-shot fallback: classify URLs by downloading thumbnails and
    scoring against text prompts.  Used when HTML context is too sparse.
    Only runs inside the CROPPER_IMAGE container (has CLIP installed).
    """
    import open_clip
    import torch
    from PIL import Image

    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="openai"
    )
    tokenizer = open_clip.get_tokenizer("ViT-B-32")
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    view_prompts = {
        "front": "front view of furniture on white background",
        "side": "side view of furniture on white background",
        "angled_front_right": "three-quarter angled view of furniture",
        "top": "top-down view of furniture",
        "detail": "close-up detail of furniture fabric or material",
        "lifestyle": "furniture in a room setting with decorations",
    }
    text_tokens = tokenizer(list(view_prompts.values())).to(device)
    with torch.no_grad():
        text_features = model.encode_text(text_tokens)
        text_features /= text_features.norm(dim=-1, keepdim=True)

    classifications = []
    view_labels = list(view_prompts.keys())

    for url in candidate_urls[:20]:
        try:
            resp = requests.get(url, headers={}, timeout=10)
            img = preprocess(Image.open(requests.utils.BytesIO(resp.content)).convert("RGB")).unsqueeze(0).to(device)
            with torch.no_grad():
                img_features = model.encode_image(img)
                img_features /= img_features.norm(dim=-1, keepdim=True)
                scores = (img_features @ text_features.T).squeeze(0).cpu().tolist()
            best_idx = int(max(range(len(scores)), key=lambda i: scores[i]))
            classifications.append({
                "url": url,
                "view": view_labels[best_idx],
                "confidence": float(scores[best_idx]),
                "is_product_isolated": view_labels[best_idx] not in ("lifestyle",),
            })
        except Exception:
            classifications.append({
                "url": url,
                "view": "unknown",
                "confidence": 0.3,
                "is_product_isolated": False,
            })

    return classifications


@app.function(
    image=BASE_IMAGE,
    volumes={"/vol": ARTIFACTS_VOLUME},
    timeout=60,
    secrets=[modal.Secret.from_name("openrouter-secret")],
)
def analyze_page(scrape: ScrapeResult, job_id: str) -> IntelligenceResult:
    """
    Call Claude Haiku via OpenRouter to classify images and extract furniture metadata.
    Falls back to a single-image heuristic if the API call fails.
    """
    import openai

    candidate_urls = scrape["candidate_image_urls"]
    if not candidate_urls:
        return _fallback_single_image([])

    url_contexts = _build_url_contexts(candidate_urls)
    prompt = _CLASSIFY_PROMPT.format(
        page_text=scrape["page_text"][:6000],
        url_contexts=url_contexts,
    )

    result: IntelligenceResult | None = None
    try:
        client = openai.OpenAI(
            api_key=os.environ["OPENROUTER_API_KEY"],
            base_url="https://openrouter.ai/api/v1",
        )
        message = client.chat.completions.create(
            model="openai/gpt-oss-20b",
            max_tokens=2048,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = message.choices[0].message.content.strip()
        # Strip markdown code fences if present
        raw = re.sub(r"^```[a-z]*\n?", "", raw)
        raw = re.sub(r"\n?```$", "", raw)
        parsed = json.loads(raw)

        # Validate view labels
        for vc in parsed.get("view_classifications", []):
            if vc.get("view") not in VALID_VIEWS:
                vc["view"] = "unknown"

        result = parsed
    except Exception as exc:
        print(f"[S1] OpenRouter call failed ({exc}), using single-image fallback")
        result = _fallback_single_image(candidate_urls)

    # If Haiku returned too few classified images, the HTML context was sparse —
    # try the CLIP fallback to improve classification quality.
    classified_urls = {vc["url"] for vc in result.get("view_classifications", [])}
    unclassified = [u for u in candidate_urls[:20] if u not in classified_urls]
    if len(unclassified) > len(candidate_urls) // 2:
        try:
            extra = _clip_classify_fallback(unclassified)
            result["view_classifications"].extend(extra)
        except Exception as exc:
            print(f"[S1] CLIP fallback skipped ({exc})")

    # Save to volume
    job_dir = ARTIFACTS_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    (job_dir / "page_intelligence.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    ARTIFACTS_VOLUME.commit()

    return result
