"""Brazilian Portuguese aesthetic content crawler."""

from __future__ import annotations

from typing import Iterator

from bs4 import BeautifulSoup
from loguru import logger

from crawler.base import BaseSource, ConsentTier, RawImagePair


class BrazilianSourcesSource(BaseSource):
    """
    Crawls Brazilian Portuguese aesthetic clinic directories and content.
    Language: Portuguese/pt-BR. NLP pairing uses Portuguese keywords from validation.yaml.
    Consent tier: 2 — clinic-posted content with implied patient consent.

    Search terms use Brazilian aesthetic vocabulary:
    - botox antes e depois (botox before and after)
    - preenchimento labial antes e depois (lip filler before and after)
    - ácido hialurônico resultado (hyaluronic acid result)
    """

    SEARCH_QUERIES_PT = [
        "botox antes e depois resultados",
        "preenchimento labial antes e depois",
        "ácido hialurônico antes e depois",
        "dysport antes e depois",
        "juvederm resultado antes depois",
        "preenchimento facial antes e depois",
    ]

    def iter_page_urls(self) -> Iterator[str]:
        import urllib.parse
        for query in self.SEARCH_QUERIES_PT:
            encoded = urllib.parse.quote(query)
            # Google Brazil search — public HTML results
            yield f"https://www.google.com.br/search?q={encoded}&tbm=isch"

    def extract_pairs_from_page(self, html: str, page_url: str) -> list[RawImagePair]:
        soup = BeautifulSoup(html, "lxml")
        pairs: list[RawImagePair] = []
        # Stub: Brazilian clinic sites have varied structures — implement per-site extractors
        logger.debug(f"[brazilian_sources] {len(pairs)} pairs from {page_url}")
        return pairs
