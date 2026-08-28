"""Clinic and practitioner website crawler."""

from __future__ import annotations

from typing import Iterator

from bs4 import BeautifulSoup
from loguru import logger

from crawler.base import BaseSource, ConsentTier, RawImagePair


class ClinicSitesSource(BaseSource):
    """
    Crawls clinic and practitioner websites for before/after galleries.
    Consent tier: 2 — practitioner-posted content implies patient consent for marketing.
    Uses a seed URL list populated via search engine discovery (see seed_discovery config).
    """

    BEFORE_PATTERNS = ["before", "pre-treatment", "pre_treatment", "pretreatment"]
    AFTER_PATTERNS = ["after", "post-treatment", "post_treatment", "posttreatment", "result"]

    def iter_page_urls(self) -> Iterator[str]:
        for url in self.config.base_urls:
            yield url

    def extract_pairs_from_page(self, html: str, page_url: str) -> list[RawImagePair]:
        soup = BeautifulSoup(html, "lxml")
        pairs: list[RawImagePair] = []

        # Strategy 1: explicit data attribute pairs
        pairs.extend(self._extract_data_attribute_pairs(soup, page_url))
        # Strategy 2: adjacent image pattern (before/after keyword in src/alt/class)
        pairs.extend(self._extract_adjacent_image_pairs(soup, page_url))

        # Deduplicate by (before_url, after_url)
        seen: set[tuple[str, str]] = set()
        unique: list[RawImagePair] = []
        for p in pairs:
            key = (p.before_url, p.after_url)
            if key not in seen:
                seen.add(key)
                unique.append(p)

        logger.debug(f"[clinic_sites] {len(unique)} pairs from {page_url}")
        return unique

    def _extract_data_attribute_pairs(self, soup, page_url: str) -> list[RawImagePair]:
        pairs: list[RawImagePair] = []
        for el in soup.select("[data-before][data-after]"):
            before_url = el.get("data-before", "").strip()
            after_url = el.get("data-after", "").strip()
            if before_url and after_url:
                pairs.append(self._make_pair(before_url, after_url, page_url))
        return pairs

    def _extract_adjacent_image_pairs(self, soup, page_url: str) -> list[RawImagePair]:
        """Find images whose src/alt/class contains before or after keywords."""
        pairs: list[RawImagePair] = []
        all_imgs = soup.find_all("img")

        before_imgs: list[str] = []
        after_imgs: list[str] = []

        for img in all_imgs:
            src = (img.get("src") or img.get("data-src") or "").lower()
            alt = (img.get("alt") or "").lower()
            cls = " ".join(img.get("class") or []).lower()
            combined = f"{src} {alt} {cls}"

            if any(p in combined for p in self.BEFORE_PATTERNS):
                url = img.get("src") or img.get("data-src") or ""
                if url:
                    before_imgs.append(url)
            elif any(p in combined for p in self.AFTER_PATTERNS):
                url = img.get("src") or img.get("data-src") or ""
                if url:
                    after_imgs.append(url)

        for b, a in zip(before_imgs, after_imgs):
            pairs.append(self._make_pair(b, a, page_url))
        return pairs

    def _make_pair(self, before_url: str, after_url: str, source_url: str) -> RawImagePair:
        return RawImagePair(
            before_url=before_url,
            after_url=after_url,
            source_url=source_url,
            source_name=self.config.name,
            language=self.config.language,
            consent_tier=ConsentTier(self.config.consent_tier),
            metadata={},
        )
