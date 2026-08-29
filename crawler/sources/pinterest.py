"""Pinterest static extraction for aesthetic before/after pins."""

from __future__ import annotations

import json
import re
from typing import Iterator
from urllib.parse import urljoin, urlparse, quote

from bs4 import BeautifulSoup
from loguru import logger

from crawler.base import BaseSource, ConsentTier, RawImagePair


IMAGE_EXT_RE = re.compile(r"\.(jpe?g|png|webp|gif)(\?.*)?$", re.IGNORECASE)
BEFORE_RE = re.compile(r"\bbefore\b|\bpre[- ]?treatment\b", re.I)
AFTER_RE = re.compile(r"\bafter\b|\bpost[- ]?treatment\b|\bresult\b", re.I)

# Pinterest board/search queries for aesthetic before/after content
SEARCH_QUERIES = [
    "botox before after results",
    "lip filler before after",
    "dermal filler before after face",
    "juvederm results before after",
    "aesthetic treatment before after",
]

PINTEREST_SEARCH_BASE = "https://www.pinterest.com/search/pins/?q={}&rs=typed"


class PinterestSource(BaseSource):
    """
    Pinterest static HTML extraction for before/after aesthetic pins.

    Pinterest search pages return JSON-LD and embedded JSON data that
    contains pin image URLs — parseable without JavaScript for search pages.

    Stage 1: Board / search pages → collect pin image groups.
    Stage 2: Individual board URLs from config → extract board pins.

    Consent tier: 3 — user-posted, no explicit consent (filtered to Tier 3).
    """

    def iter_page_urls(self) -> Iterator[str]:
        # Config-specified boards
        for url in self.config.base_urls:
            yield url

        # Search queries
        for query in SEARCH_QUERIES:
            yield PINTEREST_SEARCH_BASE.format(quote(query))

    def extract_pairs_from_page(self, html: str, page_url: str) -> list[RawImagePair]:
        soup = BeautifulSoup(html, "lxml")
        base = f"{urlparse(page_url).scheme}://{urlparse(page_url).netloc}"

        pairs: list[RawImagePair] = []

        # Method 1: JSON-LD embedded data
        pairs = self._extract_from_json_ld(soup, page_url)

        # Method 2: inline script data blobs (Pinterest embeds initial state as JSON)
        if not pairs:
            pairs = self._extract_from_inline_scripts(soup, page_url)

        # Method 3: static img tag parsing (for non-SPA fallback)
        if not pairs:
            pairs = self._extract_from_img_tags(soup, page_url, base)

        logger.debug(f"[pinterest] {len(pairs)} pairs from {page_url}")
        return pairs

    # ── JSON-LD extraction ────────────────────────────────────────────────────

    def _extract_from_json_ld(
        self, soup: BeautifulSoup, page_url: str
    ) -> list[RawImagePair]:
        pairs: list[RawImagePair] = []
        for script in soup.find_all("script", type="application/ld+json"):
            try:
                data = json.loads(script.string or "")
            except (json.JSONDecodeError, TypeError):
                continue
            if isinstance(data, list):
                items = data
            elif isinstance(data, dict):
                items = data.get("@graph", [data])
            else:
                continue

            for item in items:
                image = item.get("image") or item.get("thumbnailUrl")
                if not image:
                    continue
                name = (item.get("name") or item.get("description") or "").lower()
                if BEFORE_RE.search(name):
                    # Store as individual candidate — need a paired after image
                    logger.debug(f"[pinterest] JSON-LD before candidate: {str(image)[:60]}")

        return pairs

    # ── Inline script extraction ──────────────────────────────────────────────

    def _extract_from_inline_scripts(
        self, soup: BeautifulSoup, page_url: str
    ) -> list[RawImagePair]:
        """
        Pinterest embeds initial Redux state as JSON in a script tag.
        Extract image URLs from known JSON paths.
        """
        pairs: list[RawImagePair] = []
        before_urls: list[str] = []
        after_urls: list[str] = []

        # Find script tags with large JSON payloads (Pinterest state)
        for script in soup.find_all("script"):
            content = script.string or ""
            if "originals" not in content and "736x" not in content:
                continue
            if len(content) < 500:
                continue

            # Extract all image URLs from the JSON blob
            img_urls = re.findall(
                r'"(https://i\.pinimg\.com/[^"]+\.(?:jpg|jpeg|png|webp))"',
                content,
            )
            # Extract surrounding description text
            descriptions = re.findall(r'"description"\s*:\s*"([^"]{5,200})"', content)

            for i, url in enumerate(img_urls):
                desc = descriptions[i] if i < len(descriptions) else ""
                if BEFORE_RE.search(desc) or BEFORE_RE.search(url):
                    before_urls.append(url)
                elif AFTER_RE.search(desc) or AFTER_RE.search(url):
                    after_urls.append(url)

        for b, a in zip(before_urls, after_urls):
            pairs.append(self._make_pair(b, a, page_url, "inline_script"))

        return pairs

    # ── Static img tag fallback ───────────────────────────────────────────────

    def _extract_from_img_tags(
        self, soup: BeautifulSoup, page_url: str, base: str
    ) -> list[RawImagePair]:
        before_urls: list[str] = []
        after_urls: list[str] = []

        for img in soup.find_all("img", src=True):
            alt = (img.get("alt") or "").lower()
            src = img.get("src", "")
            if not src or not IMAGE_EXT_RE.search(src):
                continue
            if not src.startswith("http"):
                src = urljoin(base, src)

            if BEFORE_RE.search(alt) or "before" in src.lower():
                before_urls.append(src)
            elif AFTER_RE.search(alt) or "after" in src.lower():
                after_urls.append(src)

        return [
            self._make_pair(b, a, page_url, "img_tag_alt")
            for b, a in zip(before_urls, after_urls)
        ]

    # ── Helpers ───────────────────────────────────────────────────────────────

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
