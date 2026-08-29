"""Open-access publication crawler (PubMed Central, bioRxiv)."""

from __future__ import annotations

import re
from typing import Iterator
from urllib.parse import urljoin, urlparse, quote

from bs4 import BeautifulSoup
from loguru import logger

from crawler.base import BaseSource, ConsentTier, RawImagePair


IMAGE_EXT_RE = re.compile(r"\.(jpe?g|png|webp|gif|tiff?)(\?.*)?$", re.IGNORECASE)
BEFORE_RE = re.compile(r"\bbefore\b|\bpre[- ]?treatment\b|\bbaseline\b", re.I)
AFTER_RE = re.compile(r"\bafter\b|\bpost[- ]?treatment\b|\bfollow[- ]?up\b", re.I)

TREATMENT_QUERIES = [
    "botulinum toxin before after photographs aesthetic",
    "dermal filler before after clinical photographs",
    "hyaluronic acid lip augmentation results photographs",
    "injectable aesthetic treatment outcome photographs",
    "facial rejuvenation before after images",
]

PMC_SEARCH_BASE = "https://www.ncbi.nlm.nih.gov/pmc/search/?query={}&format=abstract"
BIORXIV_SEARCH_BASE = "https://www.biorxiv.org/search/{}"


class OpenAccessPubsSource(BaseSource):
    """
    Two-stage crawler for open-access academic publications.

    Stage 1: PMC/bioRxiv search results → queue article URLs.
    Stage 2: Individual article pages → extract before/after figure pairs.

    Consent tier: 1 — published under open-access license with CC terms.
    """

    def iter_page_urls(self) -> Iterator[str]:
        for url in self.config.base_urls:
            yield url
        for query in TREATMENT_QUERIES:
            encoded = quote(query)
            yield PMC_SEARCH_BASE.format(encoded)
            yield BIORXIV_SEARCH_BASE.format(encoded.replace("+", "%20"))

    def extract_pairs_from_page(self, html: str, page_url: str) -> list[RawImagePair]:
        soup = BeautifulSoup(html, "lxml")
        base = f"{urlparse(page_url).scheme}://{urlparse(page_url).netloc}"

        if "ncbi.nlm.nih.gov/pmc/search" in page_url or "biorxiv.org/search" in page_url:
            return self._process_search_results(soup, page_url, base)

        return self._process_article_page(soup, page_url, base)

    # ── Stage 1: search results ───────────────────────────────────────────────

    def _process_search_results(
        self, soup: BeautifulSoup, page_url: str, base: str
    ) -> list[RawImagePair]:
        queued = 0

        # PMC article links
        for link in soup.select(
            "a.article-title, a[href*='/pmc/articles/'], a[href*='/content/']"
        ):
            href = link.get("href", "")
            full = self._abs(href, base)
            if full and ("pmc/articles" in full or "biorxiv.org/content" in full):
                self._queue_page(full)
                queued += 1

        # bioRxiv result links
        for link in soup.select("a.highwire-cite-linked-title, span.highwire-cite-title a"):
            href = link.get("href", "")
            full = self._abs(href, base)
            if full:
                self._queue_page(full)
                queued += 1

        logger.debug(f"[open_access_pubs] Queued {queued} articles from {page_url}")
        return []

    # ── Stage 2: article pages ────────────────────────────────────────────────

    def _process_article_page(
        self, soup: BeautifulSoup, page_url: str, base: str
    ) -> list[RawImagePair]:
        pairs: list[RawImagePair] = []

        # Method 1: figure panels labelled before/after in captions
        pairs = self._pair_figures_by_caption(soup, page_url, base)

        # Method 2: adjacent supplementary figures with before/after labels
        if not pairs:
            pairs = self._pair_figures_by_alt(soup, page_url, base)

        logger.debug(f"[open_access_pubs] {len(pairs)} pairs from {page_url}")
        return pairs

    def _pair_figures_by_caption(
        self, soup: BeautifulSoup, page_url: str, base: str
    ) -> list[RawImagePair]:
        pairs: list[RawImagePair] = []
        for fig in soup.find_all("figure"):
            cap = fig.find("figcaption") or fig.find(class_=re.compile("caption", re.I))
            if not cap:
                continue
            cap_text = cap.get_text()
            if not (BEFORE_RE.search(cap_text) and AFTER_RE.search(cap_text)):
                continue
            imgs = fig.find_all("img", src=True)
            if len(imgs) >= 2:
                b = self._abs(imgs[0].get("src", ""), base)
                a = self._abs(imgs[1].get("src", ""), base)
                if b and a:
                    pairs.append(self._make_pair(b, a, page_url, "figure_caption"))
        return pairs

    def _pair_figures_by_alt(
        self, soup: BeautifulSoup, page_url: str, base: str
    ) -> list[RawImagePair]:
        before_urls: list[str] = []
        after_urls: list[str] = []
        for img in soup.find_all("img", src=True):
            alt = (img.get("alt") or "").lower()
            src = self._abs(img.get("src", ""), base)
            if not src or not IMAGE_EXT_RE.search(src):
                continue
            if BEFORE_RE.search(alt):
                before_urls.append(src)
            elif AFTER_RE.search(alt):
                after_urls.append(src)
        return [
            self._make_pair(b, a, page_url, "alt_text")
            for b, a in zip(before_urls, after_urls)
        ]

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _abs(url: str, base: str) -> str:
        if not url:
            return ""
        if url.startswith("http"):
            return url
        return urljoin(base, url)

    def _make_pair(
        self, before_url: str, after_url: str, source_url: str, method: str
    ) -> RawImagePair:
        return RawImagePair(
            before_url=before_url,
            after_url=after_url,
            source_url=source_url,
            source_name=self.config.name,
            language=self.config.language,
            consent_tier=ConsentTier(self.config.consent_tier),
            metadata={"extraction_method": method},
        )
