"""Training and academy portal crawler (Galderma Institute, Allergan Medical Institute)."""

from __future__ import annotations

from typing import Iterator

from loguru import logger

from crawler.base import BaseSource, ConsentTier, RawImagePair


class AcademyPortalsSource(BaseSource):
    """
    Crawls publicly accessible case studies and clinical galleries on
    brand training and academy portals.
    Consent tier: 1 — brand-published educational content with patient consent.
    Requires Playwright for SPA rendering.
    """

    def iter_page_urls(self) -> Iterator[str]:
        for portal in self.config.extra.get("portals", []):
            yield portal.get("base_url", "")

    def extract_pairs_from_page(self, html: str, page_url: str) -> list[RawImagePair]:
        """
        Academy portals use custom SPA layouts — Playwright renders the page,
        then this method extracts case study images.
        Stub: per-portal extractors added as each portal is mapped.
        """
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(html, "lxml")
        pairs: list[RawImagePair] = []
        logger.debug(f"[academy_portals] Stub extraction from {page_url} — {len(pairs)} pairs")
        return pairs
