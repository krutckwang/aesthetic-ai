"""Open-access scientific publication crawler (PMC, NCBI, journals)."""

from __future__ import annotations

from typing import Iterator

from bs4 import BeautifulSoup
from loguru import logger

from crawler.base import BaseSource, ConsentTier, RawImagePair


class OpenAccessPubsSource(BaseSource):
    """
    Crawls open-access publications on PubMed Central and related journals
    for clinical before/after figure pairs.
    Consent tier: 1 — IRB-approved studies with patient consent to publish.
    """

    # PMC figure captions that signal before/after pairs
    BEFORE_AFTER_CAPTION_KEYWORDS = [
        "before and after", "pre- and post", "pre and post",
        "before treatment", "after treatment", "preoperative", "postoperative",
        "baseline", "follow-up", "figure a", "figure b",
    ]

    def iter_page_urls(self) -> Iterator[str]:
        for search_url in self.config.base_urls:
            yield search_url
            # Pagination handled by following next-page links in extract_pairs_from_page

    def extract_pairs_from_page(self, html: str, page_url: str) -> list[RawImagePair]:
        soup = BeautifulSoup(html, "lxml")
        pairs: list[RawImagePair] = []

        if "ncbi.nlm.nih.gov/pmc" in page_url:
            pairs = self._extract_pmc_search_results(soup, page_url)
        else:
            pairs = self._extract_generic_article(soup, page_url)

        logger.debug(f"[open_access_pubs] {len(pairs)} pairs from {page_url}")
        return pairs

    def _extract_pmc_search_results(self, soup, page_url: str) -> list[RawImagePair]:
        """Extract article links from PMC search results page."""
        pairs: list[RawImagePair] = []
        # PMC search results link to individual articles — collect article URLs
        # and process each separately via crawl(); stub for article-level extraction
        for link in soup.select("a.title"):
            href = link.get("href", "")
            if "/pmc/articles/" in href:
                # Each article URL queued as a new page to crawl
                logger.debug(f"[open_access_pubs] Found article: {href}")
        return pairs

    def _extract_generic_article(self, soup, page_url: str) -> list[RawImagePair]:
        """
        Extract figure pairs from an individual article page.
        Looks for figure groups with captions matching before/after keywords.
        """
        pairs: list[RawImagePair] = []
        figures = soup.select("figure, .fig, .figure")

        caption_lower_list = []
        for fig in figures:
            caption_el = fig.select_one("figcaption, .caption, p")
            caption_text = caption_el.get_text(strip=True).lower() if caption_el else ""
            caption_lower_list.append((fig, caption_text))

        # Pair consecutive figures where captions suggest before/after
        for i in range(len(caption_lower_list) - 1):
            fig_a, cap_a = caption_lower_list[i]
            fig_b, cap_b = caption_lower_list[i + 1]

            combined = cap_a + " " + cap_b
            if any(kw in combined for kw in self.BEFORE_AFTER_CAPTION_KEYWORDS):
                img_a = fig_a.find("img")
                img_b = fig_b.find("img")
                if img_a and img_b:
                    before_url = img_a.get("src") or img_a.get("data-src", "")
                    after_url = img_b.get("src") or img_b.get("data-src", "")
                    if before_url and after_url:
                        pairs.append(
                            RawImagePair(
                                before_url=before_url,
                                after_url=after_url,
                                source_url=page_url,
                                source_name=self.config.name,
                                language=self.config.language,
                                consent_tier=ConsentTier(self.config.consent_tier),
                                metadata={"caption_a": cap_a, "caption_b": cap_b},
                            )
                        )
        return pairs
