"""Brand and manufacturer before/after gallery crawler (Playwright — SPA sources)."""

from __future__ import annotations

from typing import Iterator

from loguru import logger

from crawler.base import BaseSource, ConsentTier, RawImagePair


class BrandGalleriesSource(BaseSource):
    """
    Crawls before/after result galleries from major aesthetic brand websites.
    Uses Playwright headless Chromium as these are React/Angular SPAs.
    Consent tier: 1 — brand-posted content implies patient consent for marketing.

    Brands covered: Allergan, Galderma, Merz, Revance, Evolus, Solta, InMode,
    Candela, Cutera, Lumenis, Fotona, Cynosure, Teoxane.
    """

    def iter_page_urls(self) -> Iterator[str]:
        brands = self.config.extra.get("brands", [])
        for brand in brands:
            yield brand.get("base_url", "")

    def extract_pairs_from_page(self, html: str, page_url: str) -> list[RawImagePair]:
        """
        Brand gallery extraction requires Playwright rendering.
        This method receives pre-rendered HTML from the headless fetch layer.
        Each brand has different gallery structure — brand-specific selectors
        are defined in per-brand extraction methods called from here.
        """
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(html, "lxml")
        pairs: list[RawImagePair] = []

        brand_name = self._detect_brand(page_url)
        extractor = self._get_brand_extractor(brand_name)
        if extractor:
            pairs = extractor(soup, page_url)
        else:
            # Generic fallback: look for img tags with before/after patterns in src or alt
            pairs = self._generic_gallery_extract(soup, page_url)

        logger.debug(f"[brand_galleries:{brand_name}] {len(pairs)} pairs from {page_url}")
        return pairs

    def _detect_brand(self, url: str) -> str:
        brand_map = {
            "allergan": "allergan",
            "galderma": "galderma",
            "merz": "merz",
            "revance": "revance",
            "evolus": "evolus",
            "solta": "solta",
            "inmode": "inmode",
            "candela": "candela",
            "cutera": "cutera",
            "lumenis": "lumenis",
            "fotona": "fotona",
            "cynosure": "cynosure",
            "teoxane": "teoxane",
        }
        url_lower = url.lower()
        for key, name in brand_map.items():
            if key in url_lower:
                return name
        return "unknown"

    def _get_brand_extractor(self, brand_name: str):
        """Return brand-specific extraction function, or None for generic fallback."""
        extractors = {
            # Brand-specific extractors added here as each brand is implemented
        }
        return extractors.get(brand_name)

    def _generic_gallery_extract(self, soup, page_url: str) -> list[RawImagePair]:
        """Fallback: scan for img elements with before/after alt text or class names."""
        pairs: list[RawImagePair] = []
        images = soup.find_all("img")
        before_imgs = []
        after_imgs = []

        for img in images:
            alt = (img.get("alt", "") or "").lower()
            src = (img.get("src", "") or img.get("data-src", "") or "").lower()
            if "before" in alt or "before" in src:
                before_imgs.append(img.get("src") or img.get("data-src"))
            elif "after" in alt or "after" in src:
                after_imgs.append(img.get("src") or img.get("data-src"))

        for b, a in zip(before_imgs, after_imgs):
            if b and a:
                pairs.append(
                    RawImagePair(
                        before_url=b,
                        after_url=a,
                        source_url=page_url,
                        source_name=self.config.name,
                        language=self.config.language,
                        consent_tier=ConsentTier(self.config.consent_tier),
                        metadata={"extraction_method": "generic_alt_text"},
                    )
                )
        return pairs
