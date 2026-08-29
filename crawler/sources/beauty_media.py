"""Beauty media crawler (Allure, Real Self, Cosmopolitan)."""

from __future__ import annotations

import re
from typing import Iterator
from urllib.parse import urljoin, urlparse

from bs4 import BeautifulSoup
from loguru import logger

from crawler.base import BaseSource, ConsentTier, RawImagePair


IMAGE_EXT_RE = re.compile(r"\.(jpe?g|png|webp|gif)(\?.*)?$", re.IGNORECASE)
BEFORE_RE = re.compile(r"\bbefore\b|\bpre[- ]?treatment\b", re.I)
AFTER_RE = re.compile(r"\bafter\b|\bpost[- ]?treatment\b|\bresult\b", re.I)
GALLERY_LINK_RE = re.compile(
    r"(before[- ]after|transformation|results|gallery|before-and-after)", re.I
)


class BeautyMediaSource(BaseSource):
    """
    Two-stage crawler for English-language beauty editorial sites.

    Stage 1: Article listing / search pages → queue article URLs with
             before/after content signals.
    Stage 2: Individual article pages → extract before/after image pairs
             from galleries, inline figures, and slideshow items.

    Consent tier: 2 — editorial/licensed content.
    """

    def iter_page_urls(self) -> Iterator[str]:
        for url in self.config.base_urls:
            yield url

    def extract_pairs_from_page(self, html: str, page_url: str) -> list[RawImagePair]:
        soup = BeautifulSoup(html, "lxml")
        base = f"{urlparse(page_url).scheme}://{urlparse(page_url).netloc}"

        # Detect listing/category pages vs article pages
        if self._is_listing_page(soup, page_url):
            return self._process_listing_page(soup, page_url, base)

        return self._process_article_page(soup, page_url, base)

    # ── Stage 1: listing / category pages ────────────────────────────────────

    def _is_listing_page(self, soup: BeautifulSoup, page_url: str) -> bool:
        """Heuristic: listing pages have many article links, few inline figures."""
        article_links = soup.select("a[href]")
        figures = soup.find_all("figure")
        return len(article_links) > 5 and len(figures) < 3

    def _process_listing_page(
        self, soup: BeautifulSoup, page_url: str, base: str
    ) -> list[RawImagePair]:
        queued = 0
        for a in soup.find_all("a", href=True):
            href = a.get("href", "")
            text = a.get_text().lower()
            full = self._abs(href, base)
            if not full:
                continue
            # Only queue same-domain article links with before/after signals
            if urlparse(full).netloc == urlparse(page_url).netloc:
                if GALLERY_LINK_RE.search(href) or GALLERY_LINK_RE.search(text):
                    self._queue_page(full)
                    queued += 1

        # Pagination
        next_link = soup.select_one(
            "a[rel='next'], a.next, a[aria-label*='next'], a[class*='next']"
        )
        if next_link:
            href = next_link.get("href", "")
            full = self._abs(href, base)
            if full and urlparse(full).netloc == urlparse(page_url).netloc:
                self._queue_page(full)

        logger.debug(f"[beauty_media] Queued {queued} articles from {page_url}")
        return []

    # ── Stage 2: article pages ────────────────────────────────────────────────

    def _process_article_page(
        self, soup: BeautifulSoup, page_url: str, base: str
    ) -> list[RawImagePair]:
        pairs: list[RawImagePair] = []

        # Strategy 1: inline figure captions with before/after text
        pairs = self._pair_figures_by_caption(soup, page_url, base)

        # Strategy 2: slideshow items
        if not pairs:
            pairs = self._pair_slideshow_items(soup, page_url, base)

        # Strategy 3: gallery grid alt text
        if not pairs:
            pairs = self._pair_by_alt_text(soup, page_url, base)

        logger.debug(f"[beauty_media] {len(pairs)} pairs from {page_url}")
        return pairs

    def _pair_figures_by_caption(
        self, soup: BeautifulSoup, page_url: str, base: str
    ) -> list[RawImagePair]:
        pairs: list[RawImagePair] = []
        for fig in soup.find_all("figure"):
            cap = fig.find("figcaption")
            if not cap:
                continue
            cap_text = cap.get_text()
            if BEFORE_RE.search(cap_text) and AFTER_RE.search(cap_text):
                imgs = fig.find_all("img", src=True)
                if len(imgs) >= 2:
                    b = self._abs(imgs[0].get("src") or imgs[0].get("data-src", ""), base)
                    a = self._abs(imgs[1].get("src") or imgs[1].get("data-src", ""), base)
                    if b and a:
                        pairs.append(self._make_pair(b, a, page_url, "figure_caption"))
        return pairs

    def _pair_slideshow_items(
        self, soup: BeautifulSoup, page_url: str, base: str
    ) -> list[RawImagePair]:
        """
        Slides that contain a before/after pair each (single slide with 2 images).
        Also handles adjacent before/after slides.
        """
        pairs: list[RawImagePair] = []
        before_srcs: list[str] = []
        after_srcs: list[str] = []

        for slide in soup.select(
            "[class*='slide'], [class*='Slide'], li[class*='gallery'], "
            "[data-slide-index], [class*='carousel-item']"
        ):
            label = slide.get_text(strip=True).lower()
            imgs = slide.find_all("img", src=True)
            if not imgs:
                continue
            src = self._abs(imgs[0].get("src") or imgs[0].get("data-src", ""), base)
            if not src:
                continue
            if BEFORE_RE.search(label) or any(
                BEFORE_RE.search(img.get("alt") or "") for img in imgs
            ):
                before_srcs.append(src)
            elif AFTER_RE.search(label) or any(
                AFTER_RE.search(img.get("alt") or "") for img in imgs
            ):
                after_srcs.append(src)

            # Two images inside a single slide = before/after pair
            if len(imgs) >= 2:
                b = self._abs(imgs[0].get("src") or imgs[0].get("data-src", ""), base)
                a = self._abs(imgs[1].get("src") or imgs[1].get("data-src", ""), base)
                if b and a:
                    pairs.append(self._make_pair(b, a, page_url, "slide_dual"))

        for b, a in zip(before_srcs, after_srcs):
            pairs.append(self._make_pair(b, a, page_url, "slide_adjacent"))

        return pairs

    def _pair_by_alt_text(
        self, soup: BeautifulSoup, page_url: str, base: str
    ) -> list[RawImagePair]:
        before_urls: list[str] = []
        after_urls: list[str] = []
        for img in soup.find_all("img", src=True):
            alt = (img.get("alt") or "").lower()
            src = self._abs(img.get("src") or img.get("data-src", ""), base)
            if not src or not IMAGE_EXT_RE.search(src):
                continue
            if BEFORE_RE.search(alt) or "before" in src.lower():
                before_urls.append(src)
            elif AFTER_RE.search(alt) or "after" in src.lower():
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
