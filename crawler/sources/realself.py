"""RealSelf.com crawler — Tier 1 consent, calibration bootstrap source."""

from __future__ import annotations

from typing import Iterator

from bs4 import BeautifulSoup
from loguru import logger

from crawler.base import BaseSource, ConsentTier, RawImagePair, SourceConfig


class RealSelfSource(BaseSource):
    """
    Crawls RealSelf public before/after review pages.

    RealSelf enforces explicit before/after labelling at the platform level,
    making it the ground-truth source for calibration bootstrapping.
    Consent tier: 1 — users self-post their own images publicly.
    """

    def iter_page_urls(self) -> Iterator[str]:
        for base_url in self.config.base_urls:
            page = 1
            while True:
                yield f"{base_url}?page={page}"
                page += 1
                if page > 50:  # safety cap per seed URL
                    break

    def extract_pairs_from_page(self, html: str, page_url: str) -> list[RawImagePair]:
        soup = BeautifulSoup(html, "lxml")
        pairs: list[RawImagePair] = []

        # RealSelf review cards contain data-before-photo and data-after-photo attributes
        for card in soup.select("[data-before-photo][data-after-photo]"):
            before_url = card.get("data-before-photo", "").strip()
            after_url = card.get("data-after-photo", "").strip()

            if not before_url or not after_url:
                continue

            metadata = self._extract_card_metadata(card)

            pairs.append(
                RawImagePair(
                    before_url=before_url,
                    after_url=after_url,
                    source_url=page_url,
                    source_name=self.config.name,
                    language=self.config.language,
                    consent_tier=ConsentTier(self.config.consent_tier),
                    metadata=metadata,
                )
            )

        logger.debug(f"[realself] Extracted {len(pairs)} pairs from {page_url}")
        return pairs

    def _extract_card_metadata(self, card) -> dict:
        metadata: dict = {}
        treatment_el = card.select_one("[data-treatment-name]")
        if treatment_el:
            metadata["treatment_name"] = treatment_el.get("data-treatment-name", "")
        provider_el = card.select_one("[data-provider-name]")
        if provider_el:
            metadata["provider_name"] = provider_el.get("data-provider-name", "")
        date_el = card.select_one("time[datetime]")
        if date_el:
            metadata["date_posted"] = date_el.get("datetime", "")
        return metadata
