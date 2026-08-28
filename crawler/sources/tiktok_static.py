"""
TikTok static post crawler.

DISABLED BY DEFAULT: enabled: false in configs/crawler.yaml.
TikTok's bot detection is aggressive. This module handles static post pages only
(not video). Video content is handled by the deferred video_sources module.

To enable: requires proxy rotation and session management.
Not included in data volume targets.
"""

from __future__ import annotations

from typing import Iterator

from loguru import logger

from crawler.base import BaseSource, RawImagePair


class TikTokStaticSource(BaseSource):
    """TikTok static image post crawler — disabled by default."""

    def iter_page_urls(self) -> Iterator[str]:
        if self.config.enabled:
            logger.warning(
                "[tiktok_static] Source enabled without proxy rotation — expect IP bans."
            )
        return iter([])

    def extract_pairs_from_page(self, html: str, page_url: str) -> list[RawImagePair]:
        return []
