"""
S0 — Scrape
Renders the product page (static + Playwright fallback) and returns the full
candidate image URL list alongside stripped page text.  Differs from Pipeline
A's single-URL output: we return everything so S1 (VLM) can reason over it.
"""
from __future__ import annotations

import json
import re
from urllib.parse import urljoin

import modal
import requests
from bs4 import BeautifulSoup

from images import ARTIFACTS_VOLUME, ARTIFACTS_DIR, BASE_IMAGE, app
from schemas import ScrapeResult

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    )
}


# ---------------------------------------------------------------------------
# Internal helpers (pure functions — no Modal, testable locally)
# ---------------------------------------------------------------------------

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
    """Return deduplicated image URLs in rough priority order."""
    soup = BeautifulSoup(html, "lxml")
    urls: list[str] = []

    for selector in [
        'meta[property="og:image"]',
        'meta[property="og:image:url"]',
        'meta[name="twitter:image"]',
        'meta[property="twitter:image"]',
    ]:
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

    seen: set[str] = set()
    deduped: list[str] = []
    for url in urls:
        if url and url not in seen:
            seen.add(url)
            deduped.append(url)
    return deduped


def _strip_page_text(html: str) -> str:
    """Return visible text from the page, collapsing whitespace."""
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script", "style", "noscript", "svg", "path"]):
        tag.decompose()
    raw = soup.get_text(separator=" ")
    return re.sub(r"\s+", " ", raw).strip()[:8000]  # cap at 8 k chars


# ---------------------------------------------------------------------------
# Modal functions
# ---------------------------------------------------------------------------

@app.function(image=BASE_IMAGE, timeout=300)
def render_page(url: str) -> str:
    """Run Playwright to capture the fully-rendered HTML of a page."""
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


@app.function(
    image=BASE_IMAGE,
    volumes={"/vol": ARTIFACTS_VOLUME},
    timeout=120,
)
def scrape_page(page_url: str, job_id: str) -> ScrapeResult:
    """
    Fetch the product page and extract all candidate image URLs + page text.
    Tries a simple HTTP GET first; falls back to Playwright if the page
    requires JavaScript to render its image gallery.
    """
    try:
        response = requests.get(page_url, headers=HEADERS, timeout=30)
        response.raise_for_status()
        html = response.text
        urls = _extract_html_images(html, page_url)
    except Exception:
        urls = []

    if len(urls) < 3:
        # Too few images from static HTML — use the full Playwright render
        html = render_page.remote(page_url)
        urls = _extract_html_images(html, page_url)

    page_text = _strip_page_text(html)

    result: ScrapeResult = {
        "source_url": page_url,
        "page_text": page_text,
        "candidate_image_urls": urls,
    }

    # Persist to volume so other stages can reference it
    job_dir = ARTIFACTS_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    (job_dir / "page_scrape.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    ARTIFACTS_VOLUME.commit()

    return result
