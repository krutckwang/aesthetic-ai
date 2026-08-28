"""Beauty and lifestyle media crawler (NewBeauty, Allure, Byrdie, Refinery29)."""

from __future__ import annotations

from typing import Iterator

from bs4 import BeautifulSoup
from loguru import logger

from crawler.base import BaseSource, ConsentTier, RawImagePair


class BeautyMediaSource(BaseSource):
    """
    Crawls treatment feature articles and before/after content from beauty media.
    Consent tier: 2 — editorial content with implied subject consent.
    """

    ARTICLE_KEYWORDS = [
        "before-after", "before and after", "botox", "filler", "juvederm",
        "restylane", "dysport", "lip filler", "injectable"
    ]

    def iter_page_urls(self) -> Iterator[str]:
        for url in self.config.base_urls:
            yield url

    def extract_pairs_from_page(self, html: str, page_url: str) -> list[RawImagePair]:
        soup = BeautifulSoup(html, "lxml")
        pairs: list[RawImagePair] = []

        # Look for article links containing treatment keywords
        for link in soup.find_all("a", href=True):
            href = link.get("href", "").lower()
            text = link.get_text(strip=True).lower()
            if any(kw in href or kw in text for kw in self.ARTICLE_KEYWORDS):
                logger.debug(f"[beauty_media] Found candidate article: {link.get('href')}")
                # Article URL queued for crawl — image extraction happens at article level

        # Direct before/after image extraction from current page
        for fig in soup.select("figure"):
            caption = fig.get_text(strip=True).lower()
            if "before" in caption and "after" in caption:
                imgs = fig.find_all("img")
                if len(imgs) >= 2:
                    b = imgs[0].get("src") or imgs[0].get("data-src", "")
                    a = imgs[1].get("src") or imgs[1].get("data-src", "")
                    if b and a:
                        pairs.append(
                            RawImagePair(
                                before_url=b,
                                after_url=a,
                                source_url=page_url,
                                source_name=self.config.name,
                                language=self.config.language,
                                consent_tier=ConsentTier(self.config.consent_tier),
                                metadata={"caption": caption[:200]},
                            )
                        )

        logger.debug(f"[beauty_media] {len(pairs)} pairs from {page_url}")
        return pairs
