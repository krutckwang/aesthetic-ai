"""Review platform crawler (Healthgrades, Yelp, RateMDs)."""

from __future__ import annotations

from typing import Iterator

from bs4 import BeautifulSoup
from loguru import logger

from crawler.base import BaseSource, ConsentTier, RawImagePair


class ReviewPlatformsSource(BaseSource):
    """
    Crawls practitioner profile photo sections on medical review platforms.
    Consent tier: 2 — practitioner-posted marketing content.
    """

    def iter_page_urls(self) -> Iterator[str]:
        for platform in self.config.extra.get("platforms", []):
            yield platform.get("base_url", "")

    def extract_pairs_from_page(self, html: str, page_url: str) -> list[RawImagePair]:
        soup = BeautifulSoup(html, "lxml")
        pairs: list[RawImagePair] = []

        if "healthgrades" in page_url:
            pairs = self._extract_healthgrades(soup, page_url)
        elif "yelp" in page_url:
            pairs = self._extract_yelp(soup, page_url)
        elif "ratemds" in page_url:
            pairs = self._extract_ratemds(soup, page_url)

        logger.debug(f"[review_platforms] {len(pairs)} pairs from {page_url}")
        return pairs

    def _extract_healthgrades(self, soup, page_url: str) -> list[RawImagePair]:
        # Stub — Healthgrades photo galleries vary by practitioner profile
        return []

    def _extract_yelp(self, soup, page_url: str) -> list[RawImagePair]:
        # Stub — Yelp business photo sections
        return []

    def _extract_ratemds(self, soup, page_url: str) -> list[RawImagePair]:
        # Stub — RateMDs practitioner profiles
        return []
