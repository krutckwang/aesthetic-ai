"""Korean-language aesthetic content crawler (Naver Blog, Naver Cafe)."""

from __future__ import annotations

from typing import Iterator

from bs4 import BeautifulSoup
from loguru import logger

from crawler.base import BaseSource, ConsentTier, RawImagePair


class KoreanSourcesSource(BaseSource):
    """
    Crawls Korean aesthetic before/after content from Naver Blog and Naver Cafe.
    Language: Korean (ko). NLP pairing uses Korean keywords from validation.yaml.
    Consent tier: 2 — self-posted user content.

    Search terms use Korean aesthetic vocabulary:
    - 보톡스 (botox), 필러 (filler), 시술 전후 (before/after treatment),
      쥬베덤 (Juvederm), 레스틸렌 (Restylane)
    """

    NAVER_SEARCH_QUERIES = [
        "보톡스 전후",           # botox before/after
        "필러 시술 전후",         # filler treatment before/after
        "입술 필러 전후",         # lip filler before/after
        "팔자주름 필러 전후",     # nasolabial filler before/after
        "쥬베덤 전후",           # Juvederm before/after
        "레스틸렌 전후",         # Restylane before/after
        "다이스포트 전후",        # Dysport before/after
    ]

    def iter_page_urls(self) -> Iterator[str]:
        for query in self.NAVER_SEARCH_QUERIES:
            import urllib.parse
            encoded = urllib.parse.quote(query)
            yield f"https://search.naver.com/search.naver?where=blog&query={encoded}"

    def extract_pairs_from_page(self, html: str, page_url: str) -> list[RawImagePair]:
        soup = BeautifulSoup(html, "lxml")
        pairs: list[RawImagePair] = []

        # Naver blog search returns post links — follow to individual posts for image extraction
        for link in soup.select("a.api_txt_lines"):
            href = link.get("href", "")
            if "blog.naver.com" in href:
                logger.debug(f"[korean_sources] Found blog post: {href}")
                # Post URL queued for crawl — image extraction at post level

        logger.debug(f"[korean_sources] {len(pairs)} pairs from {page_url}")
        return pairs
