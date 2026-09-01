"""Professional society crawler (ASPS, AAD, AAFPRS, ISAPS, IPRAS)."""

from __future__ import annotations

import re
from typing import Iterator
from urllib.parse import urljoin, urlparse

from bs4 import BeautifulSoup
from loguru import logger

from crawler.base import BaseSource, ConsentTier, RawImagePair


IMAGE_EXT_RE = re.compile(r"\.(jpe?g|png|webp|gif)(\?.*)?$", re.IGNORECASE)
BEFORE_RE = re.compile(r"\bbefore\b|\bpre[- ]?op\b|\bbaseline\b|\bpre-treatment\b", re.I)
AFTER_RE = re.compile(
    r"\bafter\b|\bpost[- ]?op\b|\bresult\b|\boutcome\b|\bpost-treatment\b", re.I
)

# Maps URL path slugs → canonical treatment category
PROCEDURE_SLUG_MAP: dict[str, str] = {
    # Injectables
    "botulinum-toxin": "botox",
    "botox": "botox",
    "botox-cosmetic": "botox",
    "lip-augmentation": "lip_filler",
    "lip-augmentation---enhancement": "lip_filler",
    "lip-enhancement": "lip_filler",
    "lip-filler": "lip_filler",
    "dermal-fillers": "dermal_filler",
    "dermal-filler": "dermal_filler",
    "hyaluronic-acid-filler": "dermal_filler",
    "chin-augmentation": "jawline_filler",
    "chin-implants": "jawline_filler",
    "jawline-filler": "jawline_filler",
    "cheek-augmentation": "dermal_filler",
    "cheek-implants": "dermal_filler",
    "under-eye-filler": "under_eye_filler",
    "tear-trough": "under_eye_filler",
    "kybella": "kybella",
    # Surgical face
    "rhinoplasty": "rhinoplasty",
    "nose-surgery": "rhinoplasty",
    "nose-reshaping": "rhinoplasty",
    "facelift": "facelift",
    "face-lift": "facelift",
    "mini-facelift": "facelift",
    "brow-lift": "facelift",
    "forehead-lift": "facelift",
    "eyelid-surgery": "blepharoplasty",
    "blepharoplasty": "blepharoplasty",
    "upper-eyelid-surgery": "blepharoplasty",
    "lower-eyelid-surgery": "blepharoplasty",
    "neck-lift": "facelift",
    "otoplasty": "otoplasty",
    "ear-surgery": "otoplasty",
    # Skin treatments
    "laser-skin-resurfacing": "laser_resurfacing",
    "laser-resurfacing": "laser_resurfacing",
    "laser-treatment": "laser_resurfacing",
    "chemical-peel": "chemical_peel",
    "chemical-peels": "chemical_peel",
    "microneedling": "microneedling",
    "prp": "prp",
    "platelet-rich-plasma": "prp",
    "thread-lift": "thread_lift",
}

# Per-society CSS selector configs for before/after containers
SOCIETY_SELECTORS: dict[str, dict] = {
    "plasticsurgery.org": {
        "container": "[class*='before-after'], [class*='BeforeAfter'], .patient-gallery",
        "next_page": "a.next, a[rel='next']",
    },
    "aad.org": {
        "container": "[class*='before-after'], .results-gallery, .procedure-photos",
        "next_page": "a.pager-next, a[aria-label*='next']",
    },
    "aafprs.org": {
        "container": "[class*='gallery'], [class*='results'], [class*='photos']",
        "next_page": "a.next-page",
    },
    "isaps.org": {
        "container": "[class*='gallery'], .case-study-photos, .before-after",
        "next_page": "a.next",
    },
    "ipras.org": {
        "container": "[class*='gallery'], .case-results",
        "next_page": "a.next-page, a[rel='next']",
    },
}

GALLERY_URL_RE = re.compile(
    r"(photo[- ]?galleriy|before[- ]?after|patient[- ]?result|gallery|outcome|case-stud)",
    re.I,
)


class ProfessionalSocietiesSource(BaseSource):
    """
    Two-stage crawler for professional plastic surgery and dermatology society websites.

    Stage 1: Top-level procedure or news pages → queue gallery/case-study URLs.
    Stage 2: Gallery pages → extract before/after image pairs using per-society selectors.

    Consent tier: 1 — professionally curated, institution-reviewed content.
    """

    def iter_page_urls(self) -> Iterator[str]:
        for url in self.config.base_urls:
            yield url

    def extract_pairs_from_page(self, html: str, page_url: str) -> list[RawImagePair]:
        soup = BeautifulSoup(html, "lxml")
        base = f"{urlparse(page_url).scheme}://{urlparse(page_url).netloc}"
        domain = urlparse(page_url).netloc.lstrip("www.")

        selectors = self._get_selectors(domain)
        pairs = self._extract_pairs(soup, page_url, base, selectors)
        self._queue_gallery_links(soup, page_url, base)
        self._queue_next_page(soup, page_url, base, selectors)

        logger.debug(f"[professional_societies] {len(pairs)} pairs from {page_url}")
        return pairs

    # ── Pair extraction ───────────────────────────────────────────────────────

    def _extract_pairs(
        self,
        soup: BeautifulSoup,
        page_url: str,
        base: str,
        selectors: dict,
    ) -> list[RawImagePair]:
        pairs: list[RawImagePair] = []
        container_sel = selectors.get("container", "")

        # Method 1: society-specific container selectors
        if container_sel:
            for container in soup.select(container_sel):
                imgs = container.find_all("img", src=True)
                for i in range(0, len(imgs) - 1, 2):
                    b = self._abs(
                        imgs[i].get("src") or imgs[i].get("data-src", ""), base
                    )
                    a = self._abs(
                        imgs[i + 1].get("src") or imgs[i + 1].get("data-src", ""), base
                    )
                    if b and a and IMAGE_EXT_RE.search(b) and IMAGE_EXT_RE.search(a):
                        pairs.append(
                            self._make_pair(b, a, page_url, "society_container", soup)
                        )

        # Method 2: figure caption matching
        if not pairs:
            for fig in soup.find_all("figure"):
                cap = fig.find("figcaption")
                cap_text = cap.get_text() if cap else ""
                has_before = BEFORE_RE.search(cap_text)
                has_after = AFTER_RE.search(cap_text)
                if not (has_before or has_after):
                    continue
                imgs = fig.find_all("img", src=True)
                if len(imgs) >= 2:
                    b = self._abs(imgs[0].get("src", ""), base)
                    a = self._abs(imgs[1].get("src", ""), base)
                    if b and a:
                        pairs.append(
                            self._make_pair(b, a, page_url, "figure_caption", soup)
                        )

        # Method 3: alt-text labelled images
        if not pairs:
            before_urls: list[str] = []
            after_urls: list[str] = []
            for img in soup.find_all("img", src=True):
                alt = img.get("alt") or ""
                src = self._abs(img.get("src") or img.get("data-src", ""), base)
                if not src or not IMAGE_EXT_RE.search(src):
                    continue
                if BEFORE_RE.search(alt):
                    before_urls.append(src)
                elif AFTER_RE.search(alt):
                    after_urls.append(src)
            for b, a in zip(before_urls, after_urls):
                pairs.append(self._make_pair(b, a, page_url, "alt_text", soup))

        return pairs

    # ── Link queuing ──────────────────────────────────────────────────────────

    def _queue_gallery_links(
        self, soup: BeautifulSoup, page_url: str, base: str
    ) -> None:
        for a in soup.find_all("a", href=True):
            href = a.get("href", "")
            text = a.get_text().lower()
            full = self._abs(href, base)
            if not full:
                continue
            if urlparse(full).netloc != urlparse(page_url).netloc:
                continue
            if GALLERY_URL_RE.search(href) or GALLERY_URL_RE.search(text):
                self._queue_page(full)

    def _queue_next_page(
        self,
        soup: BeautifulSoup,
        page_url: str,
        base: str,
        selectors: dict,
    ) -> None:
        next_sel = selectors.get("next_page", "a[rel='next']")
        next_link = soup.select_one(next_sel)
        if next_link:
            href = next_link.get("href", "")
            full = self._abs(href, base)
            if full:
                self._queue_page(full)

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _get_selectors(domain: str) -> dict:
        for key, selectors in SOCIETY_SELECTORS.items():
            if key in domain:
                return selectors
        return {}

    @staticmethod
    def _abs(url: str, base: str) -> str:
        if not url:
            return ""
        if url.startswith("http"):
            return url
        return urljoin(base, url)

    def _extract_treatment(self, page_url: str, soup: BeautifulSoup) -> str | None:
        """Extract treatment category from URL path segments, then page title/h1."""
        path_parts = urlparse(page_url).path.strip("/").split("/")
        for part in path_parts:
            if part in PROCEDURE_SLUG_MAP:
                return PROCEDURE_SLUG_MAP[part]

        # Scan title and h1 for procedure keyword matches
        for tag in ("title", "h1"):
            el = soup.find(tag)
            if not el:
                continue
            text = el.get_text().lower()
            for slug, treatment in PROCEDURE_SLUG_MAP.items():
                if slug.replace("-", " ") in text:
                    return treatment

        return None

    def _make_pair(
        self,
        before_url: str,
        after_url: str,
        source_url: str,
        method: str,
        soup: BeautifulSoup | None = None,
    ) -> RawImagePair:
        treatment = self._extract_treatment(source_url, soup) if soup else None
        metadata: dict = {"extraction_method": method}
        if treatment:
            metadata["treatment_category"] = treatment
        return RawImagePair(
            before_url=before_url,
            after_url=after_url,
            source_url=source_url,
            source_name=self.config.name,
            language=self.config.language,
            consent_tier=ConsentTier(self.config.consent_tier),
            metadata=metadata,
        )
