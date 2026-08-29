"""Brand/manufacturer gallery crawler (Allergan, Galderma, Revance, Merz)."""

from __future__ import annotations

import re
from typing import Iterator
from urllib.parse import urljoin, urlparse

from bs4 import BeautifulSoup
from loguru import logger

from crawler.base import BaseSource, ConsentTier, RawImagePair


IMAGE_EXT_RE = re.compile(r"\.(jpe?g|png|webp|gif)(\?.*)?$", re.IGNORECASE)
BEFORE_RE = re.compile(r"\bbefore\b|\bpre[- ]?treatment\b|\bbaseline\b", re.I)
AFTER_RE = re.compile(r"\bafter\b|\bpost[- ]?treatment\b|\bresult\b", re.I)

# Per-brand selectors and gallery URL patterns
BRAND_CONFIG: dict[str, dict] = {
    "juvederm.com": {
        "gallery_path_re": r"(real[- ]?results|before[- ]?after|gallery|photos)",
        "container_sel": "[class*='result'], [class*='gallery'], [class*='before-after']",
        "img_sel": "img[src], img[data-src]",
    },
    "botox.com": {
        "gallery_path_re": r"(real[- ]?results|before[- ]?after|gallery|photos)",
        "container_sel": "[class*='result'], [class*='gallery'], [class*='before']",
        "img_sel": "img[src], img[data-src]",
    },
    "galderma.com": {
        "gallery_path_re": r"(before[- ]?after|gallery|restylane|dysport|sculptra)",
        "container_sel": "[class*='gallery'], [class*='before-after'], [class*='result']",
        "img_sel": "img[src], img[data-src]",
    },
    "revance.com": {
        "gallery_path_re": r"(before[- ]?after|gallery|daxi|results)",
        "container_sel": "[class*='gallery'], [class*='before-after']",
        "img_sel": "img[src], img[data-src]",
    },
    "merzaesthetics.com": {
        "gallery_path_re": r"(before[- ]?after|gallery|xeomin|belotero|results)",
        "container_sel": "[class*='gallery'], [class*='before'], [class*='result']",
        "img_sel": "img[src], img[data-src]",
    },
}


class BrandGalleriesSource(BaseSource):
    """
    Two-stage crawler for brand/manufacturer patient result galleries.

    Stage 1: Top-level or procedure pages → find and queue gallery/result URLs.
    Stage 2: Gallery pages → extract before/after image pairs.

    Most brand sites are SPAs; use config.rendering = HEADLESS for these.
    Consent tier: 2 — brand-published marketing content with patient consent implied.
    """

    def iter_page_urls(self) -> Iterator[str]:
        for url in self.config.base_urls:
            yield url

    def extract_pairs_from_page(self, html: str, page_url: str) -> list[RawImagePair]:
        soup = BeautifulSoup(html, "lxml")
        base = f"{urlparse(page_url).scheme}://{urlparse(page_url).netloc}"
        domain = urlparse(page_url).netloc.lstrip("www.")

        brand_cfg = self._get_brand_config(domain)
        pairs = self._extract_gallery_pairs(soup, page_url, base, brand_cfg)
        self._queue_gallery_links(soup, page_url, base, brand_cfg)

        logger.debug(f"[brand_galleries] {len(pairs)} pairs from {page_url}")
        return pairs

    # ── Pair extraction ───────────────────────────────────────────────────────

    def _extract_gallery_pairs(
        self,
        soup: BeautifulSoup,
        page_url: str,
        base: str,
        brand_cfg: dict,
    ) -> list[RawImagePair]:
        pairs: list[RawImagePair] = []
        container_sel = brand_cfg.get("container_sel", "")

        # Method 1: brand-specific container selectors
        if container_sel:
            for container in soup.select(container_sel):
                imgs = container.find_all("img")
                for i in range(0, len(imgs) - 1, 2):
                    b = self._get_src(imgs[i], base)
                    a = self._get_src(imgs[i + 1], base)
                    if b and a and IMAGE_EXT_RE.search(b) and IMAGE_EXT_RE.search(a):
                        pairs.append(
                            self._make_pair(b, a, page_url, "brand_container")
                        )

        # Method 2: before/after image slider (two overlapping images)
        if not pairs:
            for slider in soup.select(
                "[class*='slider'], [class*='comparison'], [data-component*='before-after']"
            ):
                imgs = slider.find_all("img")
                if len(imgs) >= 2:
                    b = self._get_src(imgs[0], base)
                    a = self._get_src(imgs[1], base)
                    if b and a:
                        pairs.append(self._make_pair(b, a, page_url, "slider"))

        # Method 3: generic alt text
        if not pairs:
            before_urls: list[str] = []
            after_urls: list[str] = []
            for img in soup.find_all("img"):
                alt = (img.get("alt") or "").lower()
                src = self._get_src(img, base)
                if not src or not IMAGE_EXT_RE.search(src):
                    continue
                if BEFORE_RE.search(alt) or "before" in src.lower():
                    before_urls.append(src)
                elif AFTER_RE.search(alt) or "after" in src.lower():
                    after_urls.append(src)
            for b, a in zip(before_urls, after_urls):
                pairs.append(self._make_pair(b, a, page_url, "alt_text"))

        return pairs

    # ── Link queuing ──────────────────────────────────────────────────────────

    def _queue_gallery_links(
        self,
        soup: BeautifulSoup,
        page_url: str,
        base: str,
        brand_cfg: dict,
    ) -> None:
        gallery_re_str = brand_cfg.get(
            "gallery_path_re",
            r"(before[- ]?after|gallery|results|photos)"
        )
        gallery_re = re.compile(gallery_re_str, re.I)

        for a in soup.find_all("a", href=True):
            href = a.get("href", "")
            text = a.get_text().lower()
            full = self._abs(href, base)
            if not full:
                continue
            if urlparse(full).netloc != urlparse(page_url).netloc:
                continue
            if gallery_re.search(href) or gallery_re.search(text):
                self._queue_page(full)

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _get_brand_config(domain: str) -> dict:
        for key, cfg in BRAND_CONFIG.items():
            if key in domain:
                return cfg
        return {}

    @staticmethod
    def _get_src(img, base: str) -> str:
        raw = (
            img.get("src")
            or img.get("data-src")
            or img.get("data-lazy-src")
            or img.get("data-original")
            or ""
        )
        if not raw:
            return ""
        if raw.startswith("http"):
            return raw
        return urljoin(base, raw)

    @staticmethod
    def _abs(url: str, base: str) -> str:
        if not url:
            return ""
        if url.startswith("http"):
            return url
        return urljoin(base, url)

    def _make_pair(
        self, before_url: str, after_url: str, source_url: str, method: str
    ) -> RawImagePair:
        return RawImagePair(
            before_url=before_url,
            after_url=after_url,
            source_url=source_url,
            source_name=self.config.name,
            language=self.config.language,
            consent_tier=ConsentTier(self.config.consent_tier),
            metadata={"extraction_method": method},
        )
