"""Pinterest aesthetic treatment board crawler."""

from __future__ import annotations

from typing import Iterator

from bs4 import BeautifulSoup
from loguru import logger

from crawler.base import BaseSource, ConsentTier, RawImagePair


class PinterestSource(BaseSource):
    """
    Crawls Pinterest public boards and search results for aesthetic before/after pins.
    Consent tier: 2 — publicly posted but no explicit release language.
    """

    SEARCH_QUERIES = [
        "botox before after results",
        "lip filler before after",
        "juvederm before after",
        "restylane before after",
        "dysport results before after",
        "facial filler before and after",
    ]

    def iter_page_urls(self) -> Iterator[str]:
        for query in self.SEARCH_QUERIES:
            encoded = query.replace(" ", "%20")
            yield f"https://www.pinterest.com/search/pins/?q={encoded}"

    def extract_pairs_from_page(self, html: str, page_url: str) -> list[RawImagePair]:
        """
        Pinterest renders pins via JavaScript. Static HTML returns limited content.
        Full extraction requires Playwright + scroll simulation.
        Stub implementation — override with headless rendering when activated.
        """
        soup = BeautifulSoup(html, "lxml")
        pairs: list[RawImagePair] = []
        logger.debug(f"[pinterest] Stub extraction from {page_url} — {len(pairs)} pairs")
        return pairs
