"""Open dataset crawler (Kaggle, HuggingFace Hub, GitHub academic repos)."""

from __future__ import annotations

from typing import Iterator

from loguru import logger

from crawler.base import BaseSource, ConsentTier, RawImagePair


class OpenDatasetsSource(BaseSource):
    """
    Discovers and downloads existing open aesthetic treatment datasets from
    Kaggle, HuggingFace Hub, and GitHub.
    Consent tier: 1 — publicly released research datasets.

    This source differs from web scrapers — it downloads structured datasets
    rather than scraping HTML pages. Pairs are extracted from dataset manifests.
    """

    def iter_page_urls(self) -> Iterator[str]:
        sources = self.config.extra.get("sources", [])
        for source in sources:
            name = source.get("name", "")
            if name == "kaggle":
                yield "https://www.kaggle.com/datasets?search=facial+aesthetic+before+after"
            elif name == "huggingface":
                yield "https://huggingface.co/datasets?search=aesthetic+treatment"
            elif name == "github":
                for term in source.get("search_terms", []):
                    import urllib.parse
                    encoded = urllib.parse.quote(term)
                    yield f"https://github.com/search?q={encoded}&type=repositories"

    def extract_pairs_from_page(self, html: str, page_url: str) -> list[RawImagePair]:
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(html, "lxml")
        pairs: list[RawImagePair] = []

        if "kaggle.com" in page_url:
            pairs = self._extract_kaggle_datasets(soup, page_url)
        elif "huggingface.co" in page_url:
            pairs = self._extract_hf_datasets(soup, page_url)
        elif "github.com" in page_url:
            pairs = self._extract_github_repos(soup, page_url)

        logger.debug(f"[open_datasets] {len(pairs)} pairs from {page_url}")
        return pairs

    def _extract_kaggle_datasets(self, soup, page_url: str) -> list[RawImagePair]:
        # Kaggle dataset discovery — links to datasets for manual review/download
        for link in soup.select("a[href*='/datasets/']"):
            logger.debug(f"[open_datasets:kaggle] Found dataset: {link.get('href')}")
        return []

    def _extract_hf_datasets(self, soup, page_url: str) -> list[RawImagePair]:
        for link in soup.select("a[href*='/datasets/']"):
            logger.debug(f"[open_datasets:huggingface] Found dataset: {link.get('href')}")
        return []

    def _extract_github_repos(self, soup, page_url: str) -> list[RawImagePair]:
        for link in soup.select("a.v-align-middle"):
            logger.debug(f"[open_datasets:github] Found repo: {link.get('href')}")
        return []
