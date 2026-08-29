"""Review platform crawler (Healthgrades, Yelp, RateMDs)."""

from __future__ import annotations

import re
from typing import Iterator
from urllib.parse import urljoin, urlparse

from bs4 import BeautifulSoup
from loguru import logger

from crawler.base import BaseSource, ConsentTier, RawImagePair


IMAGE_EXT_RE = re.compile(r"\.(jpe?g|png|webp|gif)(\?.*)?$", re.IGNORECASE)
BEFORE_RE = re.compile(r"\bbefore\b", re.I)
AFTER_RE = re.compile(r"\bafter\b", re.I)


class ReviewPlatformsSource(BaseSource):
    """
    Crawls practitioner profile photo sections on medical review platforms.
    Healthgrades: practitioner /photos sections.
    Yelp: business photo pages for aesthetic clinics.
    RateMDs: practitioner profile pages.
    Consent tier: 2 — practitioner-posted marketing content.
    """

    def iter_page_urls(self) -> Iterator[str]:
        for url in self.config.base_urls:
            yield url

    def extract_pairs_from_page(self, html: str, page_url: str) -> list[RawImagePair]:
        soup = BeautifulSoup(html, "lxml")
        base = f"{urlparse(page_url).scheme}://{urlparse(page_url).netloc}"

        if "healthgrades.com" in page_url:
            pairs = self._extract_healthgrades(soup, page_url, base)
        elif "yelp.com" in page_url:
            pairs = self._extract_yelp(soup, page_url, base)
        elif "ratemds.com" in page_url:
            pairs = self._extract_ratemds(soup, page_url, base)
        else:
            pairs = self._extract_generic(soup, page_url, base)

        logger.debug(f"[review_platforms] {len(pairs)} pairs from {page_url}")
        return pairs

    # ── Healthgrades ──────────────────────────────────────────────────────────

    def _extract_healthgrades(
        self, soup: BeautifulSoup, page_url: str, base: str
    ) -> list[RawImagePair]:
        pairs: list[RawImagePair] = []

        # Healthgrades before/after sections use data-testid or class patterns
        for section in soup.select(
            "[data-testid*='before-after'], .before-after-photos, "
            "[class*='BeforeAfter'], [class*='before_after']"
        ):
            imgs = section.find_all("img", src=True)
            if len(imgs) >= 2:
                b = self._abs(imgs[0].get("src") or imgs[0].get("data-src", ""), base)
                a = self._abs(imgs[1].get("src") or imgs[1].get("data-src", ""), base)
                if b and a:
                    pairs.append(self._make_pair(b, a, page_url, "healthgrades_section"))

        # Fallback: photos page — look for before/after labelled images
        if not pairs:
            pairs = self._extract_generic(soup, page_url, base)

        # Queue the /photos tab of this provider profile
        for link in soup.select("a[href*='/photos'], a[href*='before-after']"):
            href = link.get("href", "")
            full = self._abs(href, base)
            if "healthgrades.com" in full:
                self._queue_page(full)

        return pairs

    # ── Yelp ──────────────────────────────────────────────────────────────────

    def _extract_yelp(
        self, soup: BeautifulSoup, page_url: str, base: str
    ) -> list[RawImagePair]:
        pairs: list[RawImagePair] = []

        # Yelp photo grids: look for photo items with before/after captions
        for item in soup.select(
            "[class*='photo-box'], [class*='photoGrid'], li[class*='photo']"
        ):
            caption = item.get_text(strip=True).lower()
            img = item.find("img", src=True)
            if img and ("before" in caption or "after" in caption):
                src = self._abs(img.get("src") or img.get("data-src", ""), base)
                if src:
                    # Store individually and pair with next matching opposite
                    logger.debug(f"[review_platforms:yelp] Candidate photo: {src[:60]}")

        # Yelp before/after specific sections
        for section in soup.select("[class*='beforeAfter'], [data-photo-type='before-after']"):
            imgs = section.find_all("img", src=True)
            if len(imgs) >= 2:
                b = self._abs(imgs[0].get("src", ""), base)
                a = self._abs(imgs[1].get("src", ""), base)
                if b and a:
                    pairs.append(self._make_pair(b, a, page_url, "yelp_before_after"))

        # Queue photo pages
        for link in soup.select("a[href*='/photos']"):
            href = link.get("href", "")
            full = self._abs(href, base)
            if "yelp.com" in full:
                self._queue_page(full)

        return pairs

    # ── RateMDs ───────────────────────────────────────────────────────────────

    def _extract_ratemds(
        self, soup: BeautifulSoup, page_url: str, base: str
    ) -> list[RawImagePair]:
        pairs: list[RawImagePair] = []

        # RateMDs practitioner profiles may have photo galleries
        for gallery in soup.select(
            ".photo-gallery, .profile-photos, [class*='gallery']"
        ):
            imgs = gallery.find_all("img", src=True)
            before_imgs = []
            after_imgs = []
            for img in imgs:
                alt = (img.get("alt") or "").lower()
                src = self._abs(img.get("src") or img.get("data-src", ""), base)
                if not src:
                    continue
                if BEFORE_RE.search(alt):
                    before_imgs.append(src)
                elif AFTER_RE.search(alt):
                    after_imgs.append(src)

            for b, a in zip(before_imgs, after_imgs):
                pairs.append(self._make_pair(b, a, page_url, "ratemds_gallery"))

        return pairs

    # ── Generic fallback ──────────────────────────────────────────────────────

    def _extract_generic(
        self, soup: BeautifulSoup, page_url: str, base: str
    ) -> list[RawImagePair]:
        """Generic before/after image pair extraction."""
        before_imgs: list[str] = []
        after_imgs: list[str] = []

        for img in soup.find_all("img", src=True):
            alt = (img.get("alt") or "").lower()
            src_raw = img.get("src") or img.get("data-src", "")
            src = self._abs(src_raw, base)
            if not src or not IMAGE_EXT_RE.search(src):
                continue
            if BEFORE_RE.search(alt) or "before" in src.lower():
                before_imgs.append(src)
            elif AFTER_RE.search(alt) or "after" in src.lower():
                after_imgs.append(src)

        return [
            self._make_pair(b, a, page_url, "generic_alt")
            for b, a in zip(before_imgs, after_imgs)
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
