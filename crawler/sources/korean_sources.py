"""Korean-language aesthetic content crawler (Naver Blog, Naver Cafe)."""

from __future__ import annotations

import re
from typing import Iterator
from urllib.parse import urljoin, urlparse, quote

from bs4 import BeautifulSoup
from loguru import logger

from crawler.base import BaseSource, ConsentTier, RawImagePair


# Korean keywords indicating before/after content in blog post titles/bodies
BEFORE_KW = frozenset(["전", "시술전", "치료전", "수술전", "이전", "before"])
AFTER_KW = frozenset(["후", "시술후", "치료후", "수술후", "이후", "결과", "after"])
TREATMENT_KW = frozenset(["보톡스", "필러", "쥬베덤", "레스틸렌", "다이스포트", "입술", "팔자"])

IMAGE_EXT_RE = re.compile(r"\.(jpe?g|png|webp|gif)(\?.*)?$", re.IGNORECASE)

NAVER_SEARCH_QUERIES = [
    "보톡스 시술 전후 사진",
    "필러 시술 전후 사진",
    "입술 필러 전후 사진",
    "팔자주름 필러 전후",
    "쥬베덤 전후 사진",
    "레스틸렌 전후 사진",
    "다이스포트 보톡스 전후",
]


class KoreanSourcesSource(BaseSource):
    """
    Two-stage crawler for Korean aesthetic content.

    Stage 1: Naver Blog search results → collect blog post URLs.
    Stage 2: Individual Naver Blog posts → extract adjacent before/after images.

    Naver Blog uses a frame-based architecture; the mobile endpoint
    (m.blog.naver.com) returns flat HTML accessible via static fetch.
    Consent tier: 2 — self-posted user content.
    """

    def iter_page_urls(self) -> Iterator[str]:
        for query in NAVER_SEARCH_QUERIES:
            encoded = quote(query)
            yield (
                f"https://search.naver.com/search.naver"
                f"?where=blog&query={encoded}&sm=tab_opt&nso=so%3Ar%2Ca%3Aall"
            )

    def extract_pairs_from_page(self, html: str, page_url: str) -> list[RawImagePair]:
        soup = BeautifulSoup(html, "lxml")

        if "search.naver.com" in page_url:
            return self._process_search_results(soup, page_url)

        if "blog.naver.com" in page_url or "m.blog.naver.com" in page_url:
            return self._process_blog_post(soup, page_url)

        return []

    # ── Stage 1: Naver search results ─────────────────────────────────────────

    def _process_search_results(
        self, soup: BeautifulSoup, page_url: str
    ) -> list[RawImagePair]:
        """Collect blog post URLs and queue them for stage-2 extraction."""
        queued = 0

        # Naver blog search result links
        for link in soup.select("a.api_txt_lines, a.total_tit, a[class*='title']"):
            href = link.get("href", "")
            if not href:
                continue
            if "blog.naver.com" in href:
                mobile_url = self._to_mobile_url(href)
                self._queue_page(mobile_url)
                queued += 1

        # Also pick up links from result snippets
        for a in soup.select("div.total_wrap a[href*='blog.naver.com']"):
            href = a.get("href", "")
            if href:
                self._queue_page(self._to_mobile_url(href))
                queued += 1

        # Pagination: next search result page
        next_page = soup.select_one("a.btn_next, a[class*='next']")
        if next_page:
            next_href = next_page.get("href", "")
            if next_href:
                next_full = urljoin("https://search.naver.com", next_href)
                self._queue_page(next_full)

        logger.debug(f"[korean_sources] Queued {queued} blog posts from {page_url}")
        return []

    # ── Stage 2: Naver blog post ───────────────────────────────────────────────

    def _process_blog_post(
        self, soup: BeautifulSoup, page_url: str
    ) -> list[RawImagePair]:
        """Extract before/after image pairs from a Naver Blog post."""
        pairs: list[RawImagePair] = []

        # Collect all images in the post body
        body = soup.select_one(
            "div.se-main-container, div#postViewArea, div.post-view, div.entry-content"
        )
        if body is None:
            body = soup

        all_imgs: list[str] = []
        for img in body.find_all("img", src=True):
            src = self._resolve_naver_img_src(img)
            if src and IMAGE_EXT_RE.search(src):
                all_imgs.append(src)

        if len(all_imgs) < 2:
            return []

        # Strategy 1: look for explicit before/after alt text
        pairs = self._pair_by_alt_text(body, page_url)

        # Strategy 2: pair by adjacent caption text
        if not pairs:
            pairs = self._pair_by_caption_proximity(body, page_url)

        # Strategy 3: fallback — pair first half with second half
        if not pairs and len(all_imgs) >= 2:
            mid = len(all_imgs) // 2
            for b, a in zip(all_imgs[:mid], all_imgs[mid:]):
                # Validate with basic title keyword check
                title = soup.title.get_text() if soup.title else ""
                if any(kw in title for kw in TREATMENT_KW):
                    pairs.append(self._make_pair(b, a, page_url, "fallback_halves"))
                    if len(pairs) >= 3:
                        break

        logger.debug(f"[korean_sources] {len(pairs)} pairs from {page_url}")
        return pairs

    def _pair_by_alt_text(
        self, body: BeautifulSoup, page_url: str
    ) -> list[RawImagePair]:
        before_urls: list[str] = []
        after_urls: list[str] = []
        for img in body.find_all("img", src=True):
            alt = (img.get("alt") or "").strip()
            src = self._resolve_naver_img_src(img)
            if not src:
                continue
            if any(kw in alt for kw in BEFORE_KW):
                before_urls.append(src)
            elif any(kw in alt for kw in AFTER_KW):
                after_urls.append(src)
        return [
            self._make_pair(b, a, page_url, "alt_text")
            for b, a in zip(before_urls, after_urls)
        ]

    def _pair_by_caption_proximity(
        self, body: BeautifulSoup, page_url: str
    ) -> list[RawImagePair]:
        """Pair images where adjacent text contains before/after keywords."""
        pairs: list[RawImagePair] = []
        candidates: list[tuple[str, str]] = []  # (label, img_src)

        for el in body.descendants:
            from bs4 import Tag, NavigableString
            if isinstance(el, Tag) and el.name == "img":
                src = self._resolve_naver_img_src(el)
                if src:
                    # Look at surrounding text
                    nearby_text = ""
                    for sib in list(el.next_siblings)[:3]:
                        if isinstance(sib, NavigableString):
                            nearby_text += str(sib)
                        elif isinstance(sib, Tag):
                            nearby_text += sib.get_text()

                    if any(kw in nearby_text for kw in BEFORE_KW):
                        candidates.append(("before", src))
                    elif any(kw in nearby_text for kw in AFTER_KW):
                        candidates.append(("after", src))

        before_srcs = [s for label, s in candidates if label == "before"]
        after_srcs = [s for label, s in candidates if label == "after"]
        for b, a in zip(before_srcs, after_srcs):
            pairs.append(self._make_pair(b, a, page_url, "caption_proximity"))
        return pairs

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _to_mobile_url(url: str) -> str:
        """Convert desktop Naver blog URL to mobile version for flat HTML."""
        return url.replace("blog.naver.com", "m.blog.naver.com")

    @staticmethod
    def _resolve_naver_img_src(img) -> str:
        """Resolve Naver lazy-loaded image sources."""
        src = (
            img.get("src")
            or img.get("data-lazy-src")
            or img.get("data-src")
            or img.get("data-original")
            or ""
        )
        # Remove Naver CDN resize parameters to get original
        if "postfiles.pstatic.net" in src or "blogfiles.pstatic.net" in src:
            src = src.split("?")[0]
        return src.strip()

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
