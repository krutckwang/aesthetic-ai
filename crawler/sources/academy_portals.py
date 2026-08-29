"""Medical academy and society portal crawler (ASPS, AAFPRS, ISAPS)."""

from __future__ import annotations

import re
from typing import Iterator
from urllib.parse import urljoin, urlparse

from bs4 import BeautifulSoup
from loguru import logger

from crawler.base import BaseSource, ConsentTier, RawImagePair


IMAGE_EXT_RE = re.compile(r"\.(jpe?g|png|webp|gif)(\?.*)?$", re.IGNORECASE)
BEFORE_RE = re.compile(r"\bbefore\b|\bpre[- ]?op\b|\bbaseline\b", re.I)
AFTER_RE = re.compile(r"\bafter\b|\bpost[- ]?op\b|\bresult\b|\boutcome\b", re.I)

# URL patterns that suggest case study / photo gallery pages
CASE_STUDY_URL_RE = re.compile(
    r"(case[- ]?stud|photo[- ]?galleriy|before[- ]after|patient[- ]result|"
    r"procedure[- ]result|gallery|outcome)",
    re.I,
)


class AcademyPortalsSource(BaseSource):
    """
    Two-stage crawler for medical academy and professional portal websites.

    These sites are typically SPAs or server-rendered sites with case study
    photo galleries. Uses headless rendering (config.rendering = HEADLESS)
    for JS-heavy portals; falls back to static fetch otherwise.

    Stage 1: Landing / procedure pages → queue case study / gallery URLs.
    Stage 2: Case study pages → extract before/after figure pairs.

    Consent tier: 1 — professionally published, institution-reviewed content.
    """

    def iter_page_urls(self) -> Iterator[str]:
        for url in self.config.base_urls:
            yield url

    def extract_pairs_from_page(self, html: str, page_url: str) -> list[RawImagePair]:
        soup = BeautifulSoup(html, "lxml")
        base = f"{urlparse(page_url).scheme}://{urlparse(page_url).netloc}"

        pairs = self._extract_case_study_pairs(soup, page_url, base)
        self._queue_case_study_links(soup, page_url, base)

        logger.debug(f"[academy_portals] {len(pairs)} pairs from {page_url}")
        return pairs

    # ── Pair extraction ───────────────────────────────────────────────────────

    def _extract_case_study_pairs(
        self, soup: BeautifulSoup, page_url: str, base: str
    ) -> list[RawImagePair]:
        pairs: list[RawImagePair] = []

        # Strategy 1: explicit before/after section containers
        for container in soup.select(
            "[class*='before-after'], [class*='BeforeAfter'], "
            "[data-type='before-after'], [class*='result-photos'], "
            "[class*='case-photos'], [id*='before-after']"
        ):
            imgs = container.find_all("img", src=True)
            if len(imgs) >= 2:
                for i in range(0, len(imgs) - 1, 2):
                    b = self._abs(
                        imgs[i].get("src") or imgs[i].get("data-src", ""), base
                    )
                    a = self._abs(
                        imgs[i + 1].get("src") or imgs[i + 1].get("data-src", ""), base
                    )
                    if b and a:
                        pairs.append(self._make_pair(b, a, page_url, "section_container"))

        # Strategy 2: figure pairs by caption keywords
        if not pairs:
            for fig in soup.find_all("figure"):
                cap = fig.find("figcaption")
                cap_text = cap.get_text() if cap else ""
                if not (BEFORE_RE.search(cap_text) or AFTER_RE.search(cap_text)):
                    continue
                imgs = fig.find_all("img", src=True)
                if len(imgs) >= 2:
                    b = self._abs(imgs[0].get("src", ""), base)
                    a = self._abs(imgs[1].get("src", ""), base)
                    if b and a:
                        pairs.append(self._make_pair(b, a, page_url, "figure_caption"))

        # Strategy 3: alt-text labelled images across the whole page
        if not pairs:
            before_urls: list[str] = []
            after_urls: list[str] = []
            for img in soup.find_all("img", src=True):
                alt = (img.get("alt") or "").lower()
                src = self._abs(img.get("src") or img.get("data-src", ""), base)
                if not src or not IMAGE_EXT_RE.search(src):
                    continue
                if BEFORE_RE.search(alt):
                    before_urls.append(src)
                elif AFTER_RE.search(alt):
                    after_urls.append(src)
            for b, a in zip(before_urls, after_urls):
                pairs.append(self._make_pair(b, a, page_url, "alt_text"))

        return pairs

    # ── Link queuing ──────────────────────────────────────────────────────────

    def _queue_case_study_links(
        self, soup: BeautifulSoup, page_url: str, base: str
    ) -> None:
        """Queue internal links that point to case study or photo gallery pages."""
        for a in soup.find_all("a", href=True):
            href = a.get("href", "")
            text = a.get_text().lower()
            full = self._abs(href, base)
            if not full:
                continue
            # Only follow same-domain links
            if urlparse(full).netloc != urlparse(page_url).netloc:
                continue
            if CASE_STUDY_URL_RE.search(href) or CASE_STUDY_URL_RE.search(text):
                self._queue_page(full)

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
