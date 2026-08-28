"""Professional society website crawler (ASPS, AAD, ASDS, BAAPS, ISAPS)."""

from __future__ import annotations

from typing import Iterator

from bs4 import BeautifulSoup
from loguru import logger

from crawler.base import BaseSource, ConsentTier, RawImagePair


class ProfessionalSocietiesSource(BaseSource):
    """
    Crawls patient education and case study galleries on professional society sites.
    Consent tier: 1 — peer-reviewed/society-published content with implied consent.
    """

    def iter_page_urls(self) -> Iterator[str]:
        for url in self.config.base_urls:
            yield url

    def extract_pairs_from_page(self, html: str, page_url: str) -> list[RawImagePair]:
        soup = BeautifulSoup(html, "lxml")
        pairs: list[RawImagePair] = []
        # Generic extraction: look for before/after in image context
        for section in soup.select(".before-after, .case-study, .gallery-item, .result-image"):
            imgs = section.find_all("img")
            if len(imgs) >= 2:
                before_url = imgs[0].get("src") or imgs[0].get("data-src", "")
                after_url = imgs[1].get("src") or imgs[1].get("data-src", "")
                if before_url and after_url:
                    pairs.append(
                        RawImagePair(
                            before_url=before_url,
                            after_url=after_url,
                            source_url=page_url,
                            source_name=self.config.name,
                            language=self.config.language,
                            consent_tier=ConsentTier(self.config.consent_tier),
                            metadata={},
                        )
                    )
        logger.debug(f"[professional_societies] {len(pairs)} pairs from {page_url}")
        return pairs
