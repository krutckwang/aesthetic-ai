"""
Layer 1 — Structural heuristics validator.

Detects explicit before/after labelling in HTML page structure:
  - data-* attributes on image elements or containers
  - Image filename patterns (before_001.jpg, after_shot.png)
  - Alt text containing before/after language
  - CSS class names on image containers
  - Caption text in sibling or parent elements
  - Carousel/gallery position (weak signal — not sufficient for Gate 1 alone)

Returns a StructuralResult with:
  - has_explicit_label: True only when a strong structural signal is found.
                        Gate 1 of ordering validation requires this to be True.
  - confidence: 0.0 – 1.0 composite score
  - signals: list of signal names found (for auditing)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from urllib.parse import urlparse

from bs4 import BeautifulSoup, Tag


# ── Keyword sets (all lower-case, checked with `in`) ──────────────────────────

BEFORE_KEYWORDS: dict[str, list[str]] = {
    "en": ["before", "pre-treatment", "pre treatment", "pretreatment",
           "prior to", "untreated", "baseline", "pre-op", "preop"],
    "ko": ["전", "시술전", "치료전", "이전", "수술전", "before"],
    "pt": ["antes", "pré-tratamento", "pre-tratamento", "antes do",
           "anterior", "pre tratamento"],
}

AFTER_KEYWORDS: dict[str, list[str]] = {
    "en": ["after", "post-treatment", "post treatment", "posttreatment",
           "results", "outcome", "following", "post-op", "postop",
           "result", "after care"],
    "ko": ["후", "시술후", "치료후", "이후", "결과", "수술후", "after"],
    "pt": ["depois", "após", "pós-tratamento", "pos-tratamento",
           "resultado", "resultado final", "pós", "pos tratamento"],
}

# Flattened sets for fast lookup
_ALL_BEFORE: frozenset[str] = frozenset(
    kw for kws in BEFORE_KEYWORDS.values() for kw in kws
)
_ALL_AFTER: frozenset[str] = frozenset(
    kw for kws in AFTER_KEYWORDS.values() for kw in kws
)


@dataclass
class StructuralResult:
    """Result of Layer 1 structural heuristic analysis for one image pair."""

    has_explicit_label: bool  # True = Gate 1 passed
    confidence: float         # composite score 0.0–1.0
    before_label_signals: list[str] = field(default_factory=list)
    after_label_signals: list[str] = field(default_factory=list)
    before_ordering: str = "unknown"  # "before" | "after" | "unknown"
    after_ordering: str = "unknown"


class StructuralValidator:
    """
    Analyses HTML context of two candidate images and determines whether
    explicit before/after structural signals are present.

    Call `validate(before_url, after_url, page_html)` to get a StructuralResult.
    """

    def validate(
        self,
        before_url: str,
        after_url: str,
        page_html: str,
    ) -> StructuralResult:
        """
        Analyse page HTML and both image URLs for structural before/after signals.

        Args:
            before_url:  Candidate before image URL.
            after_url:   Candidate after image URL.
            page_html:   Full rendered HTML of the source page.

        Returns:
            StructuralResult indicating whether explicit labelling was found.
        """
        soup = BeautifulSoup(page_html, "lxml")
        before_signals = self._signals_for_url(before_url, soup, is_before=True)
        after_signals = self._signals_for_url(after_url, soup, is_before=False)

        # An explicit label requires at least one strong signal on each image
        strong_before = any(s.startswith("strong:") for s in before_signals)
        strong_after = any(s.startswith("strong:") for s in after_signals)
        has_explicit = strong_before and strong_after

        # Score: strong signals count more
        score = self._compute_score(before_signals, after_signals)

        return StructuralResult(
            has_explicit_label=has_explicit,
            confidence=score,
            before_label_signals=before_signals,
            after_label_signals=after_signals,
            before_ordering="before" if before_signals else "unknown",
            after_ordering="after" if after_signals else "unknown",
        )

    # ── Per-image signal extraction ───────────────────────────────────────────

    def _signals_for_url(
        self,
        url: str,
        soup: BeautifulSoup,
        is_before: bool,
    ) -> list[str]:
        """Collect all structural signals for a single image URL."""
        signals: list[str] = []
        expected_kw = _ALL_BEFORE if is_before else _ALL_AFTER
        img_tag = self._find_img_tag(url, soup)

        # ── Filename / URL path signal ──
        filename = urlparse(url).path.lower()
        if any(kw in filename for kw in expected_kw):
            signals.append(f"strong:filename:{filename}")

        if img_tag is None:
            return signals

        # ── data-* attribute signal ──
        for attr, val in img_tag.attrs.items():
            if isinstance(val, str):
                val_lower = val.lower()
                attr_lower = attr.lower()
                if any(kw in val_lower or kw in attr_lower for kw in expected_kw):
                    signals.append(f"strong:data_attr:{attr}={val[:40]}")

        # ── alt text signal ──
        alt = (img_tag.get("alt") or "").lower()
        if any(kw in alt for kw in expected_kw):
            signals.append(f"strong:alt_text:{alt[:60]}")

        # ── CSS class signal ──
        classes = " ".join(img_tag.get("class") or []).lower()
        if any(kw in classes for kw in expected_kw):
            signals.append(f"strong:css_class:{classes[:60]}")

        # ── Parent container class / data-* signals ──
        for parent in img_tag.parents:
            if not isinstance(parent, Tag):
                continue
            parent_classes = " ".join(parent.get("class") or []).lower()
            for attr, val in parent.attrs.items():
                if isinstance(val, str):
                    combined = (val + " " + parent_classes).lower()
                    if any(kw in combined for kw in expected_kw):
                        signals.append(f"strong:parent_attr:{attr}={val[:40]}")
                        break
            if len(list(img_tag.parents)) > 5:
                break  # don't traverse too far up

        # ── Caption text signal (figcaption, nearby <p>) ──
        caption = self._get_caption_text(img_tag)
        if any(kw in caption for kw in expected_kw):
            signals.append(f"strong:caption:{caption[:80]}")

        # ── Position in sibling group (weak signal) ──
        position = self._sibling_position(img_tag, is_before)
        if position:
            signals.append(f"weak:position:{position}")

        return signals

    # ── HTML traversal helpers ────────────────────────────────────────────────

    def _find_img_tag(self, url: str, soup: BeautifulSoup) -> Tag | None:
        """Find the <img> element whose src or data-src matches the URL."""
        filename = urlparse(url).path.split("/")[-1]
        for img in soup.find_all("img"):
            src = img.get("src") or img.get("data-src") or ""
            if url in src or (filename and filename in src):
                return img
        return None

    def _get_caption_text(self, img_tag: Tag) -> str:
        """Extract caption text from figcaption or nearby paragraph."""
        # Check immediate parent figure
        parent = img_tag.find_parent("figure")
        if parent:
            cap = parent.find("figcaption")
            if cap:
                return cap.get_text(strip=True).lower()

        # Check next sibling <p> or <span>
        for sibling in img_tag.next_siblings:
            if isinstance(sibling, Tag) and sibling.name in ("p", "span", "div", "caption"):
                return sibling.get_text(strip=True).lower()

        return ""

    def _sibling_position(self, img_tag: Tag, is_before: bool) -> str | None:
        """
        Weak signal: if image is the first in a two-image group, it may be 'before'.
        Only used as a tiebreaker, never as an explicit label.
        """
        parent = img_tag.parent
        if parent is None:
            return None
        sibling_imgs = parent.find_all("img")
        if len(sibling_imgs) == 2:
            if is_before and sibling_imgs[0] == img_tag:
                return "first_of_two"
            if not is_before and sibling_imgs[1] == img_tag:
                return "second_of_two"
        return None

    # ── Score calculation ─────────────────────────────────────────────────────

    @staticmethod
    def _compute_score(
        before_signals: list[str],
        after_signals: list[str],
    ) -> float:
        """Composite confidence score: strong signals contribute 0.45 each, weak 0.1."""
        score = 0.0
        for signals in (before_signals, after_signals):
            has_strong = any(s.startswith("strong:") for s in signals)
            has_weak = any(s.startswith("weak:") for s in signals)
            if has_strong:
                score += 0.45
            elif has_weak:
                score += 0.1
        return round(min(score, 1.0), 3)
