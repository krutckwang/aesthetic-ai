"""
Instagram public post crawler.

DISABLED BY DEFAULT: enabled: false in configs/crawler.yaml.
Instagram's bot detection will IP-ban without proxy rotation.
This module is scaffolded for future use only — not in data volume targets.

To enable: set enabled: true in configs/crawler.yaml and configure proxy
rotation. Requires session cookie injection or rotating residential proxies.
"""

from __future__ import annotations

from typing import Iterator

from loguru import logger

from crawler.base import BaseSource, RawImagePair, SourceConfig


class InstagramSource(BaseSource):
    """
    Instagram hashtag-based before/after image crawler.
    Disabled by default — see module docstring.
    """

    def iter_page_urls(self) -> Iterator[str]:
        if self.config.enabled:
            logger.warning(
                "[instagram] Source enabled without proxy rotation — expect IP bans."
            )
        return iter([])  # no-op when disabled

    def extract_pairs_from_page(self, html: str, page_url: str) -> list[RawImagePair]:
        # Stub — implementation requires carousel detection and caption NLP
        return []
