"""
Layer 2 — Multilingual NLP pairing validator.

Uses sentence-transformers (paraphrase-multilingual-MiniLM-L12-v2) to score
whether the surrounding text of two candidate images confirms a before/after
treatment relationship.

Two-stage approach:
  1. Fast path: keyword matching in extracted text (no model call needed).
  2. Slow path: sentence transformer semantic similarity with reference templates.

Supports: English, Korean, Brazilian Portuguese (priority languages).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Sequence

import numpy as np
from loguru import logger


# ── Reference template sentences ─────────────────────────────────────────────
# Used to compute semantic similarity with surrounding image text.

BEFORE_TEMPLATES: dict[str, str] = {
    "en": "This photograph shows the patient before the aesthetic treatment was performed.",
    "ko": "이 사진은 시술 전의 모습입니다.",
    "pt": "Esta fotografia mostra o paciente antes do tratamento estético.",
}

AFTER_TEMPLATES: dict[str, str] = {
    "en": "This photograph shows the patient after the aesthetic treatment results.",
    "ko": "이 사진은 시술 후 결과를 보여줍니다.",
    "pt": "Esta fotografia mostra os resultados do paciente após o tratamento estético.",
}

# ── Keyword signals for fast path ─────────────────────────────────────────────

BEFORE_KEYWORDS_FLAT: frozenset[str] = frozenset([
    "before", "pre-treatment", "pretreatment", "prior to", "untreated",
    "baseline", "전", "시술전", "치료전", "이전", "수술전",
    "antes", "pré-tratamento", "anterior",
])

AFTER_KEYWORDS_FLAT: frozenset[str] = frozenset([
    "after", "post-treatment", "posttreatment", "results", "outcome",
    "following", "post-op",
    "후", "시술후", "치료후", "이후", "결과", "수술후",
    "depois", "após", "pós-tratamento", "resultado",
])

# Score awarded for a keyword match (skips model call)
KEYWORD_MATCH_SCORE = 0.90


@dataclass
class NLPPairingResult:
    """Result of Layer 2 NLP pairing validation."""

    pairing_score: float          # 0.0–1.0
    detected_language: str        # "en" | "ko" | "pt" | "unknown"
    method: str                   # "keyword" | "semantic" | "none"
    before_text_snippet: str      # first 100 chars of extracted before-side text
    after_text_snippet: str       # first 100 chars of extracted after-side text


@lru_cache(maxsize=1)
def _load_model():
    """Load the multilingual sentence transformer once and cache it."""
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(
            "paraphrase-multilingual-MiniLM-L12-v2",
        )
        logger.info("[nlp_pairing] Sentence transformer loaded.")
        return model
    except ImportError:
        logger.warning("[nlp_pairing] sentence-transformers not installed — semantic path disabled.")
        return None


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two 1-D vectors."""
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def _detect_language(text: str) -> str:
    """Heuristic language detection from character ranges."""
    # Korean Unicode range
    if re.search(r"[가-힣ᄀ-ᇿ]", text):
        return "ko"
    # Portuguese-specific characters
    if re.search(r"[ãõáéíóúâêîôûç]", text, re.IGNORECASE):
        return "pt"
    return "en"


class NLPPairingValidator:
    """
    Validates whether surrounding text of two images confirms a before/after pair.

    Inputs are plain-text snippets extracted from the page around each image
    (captions, alt text, parent element text, aria-labels).
    """

    def __init__(self, min_keyword_score: float = KEYWORD_MATCH_SCORE) -> None:
        self.min_keyword_score = min_keyword_score

    def validate(
        self,
        before_text: str,
        after_text: str,
    ) -> NLPPairingResult:
        """
        Compute pairing score from text extracted near each image.

        Args:
            before_text: Text extracted from the context of the 'before' candidate.
            after_text:  Text extracted from the context of the 'after' candidate.

        Returns:
            NLPPairingResult with pairing_score in [0, 1].
        """
        before_lower = before_text.lower()
        after_lower = after_text.lower()
        lang = _detect_language(before_text + after_text)

        # ── Fast path: keyword matching ────────────────────────────────────────
        before_hit = any(kw in before_lower for kw in BEFORE_KEYWORDS_FLAT)
        after_hit = any(kw in after_lower for kw in AFTER_KEYWORDS_FLAT)

        if before_hit and after_hit:
            return NLPPairingResult(
                pairing_score=self.min_keyword_score,
                detected_language=lang,
                method="keyword",
                before_text_snippet=before_text[:100],
                after_text_snippet=after_text[:100],
            )

        if not before_text.strip() and not after_text.strip():
            return NLPPairingResult(
                pairing_score=0.0,
                detected_language=lang,
                method="none",
                before_text_snippet="",
                after_text_snippet="",
            )

        # ── Slow path: semantic similarity ─────────────────────────────────────
        model = _load_model()
        if model is None:
            # Fall back to partial keyword score if model unavailable
            partial = (0.5 if before_hit else 0.0) + (0.5 if after_hit else 0.0)
            return NLPPairingResult(
                pairing_score=partial * 0.6,
                detected_language=lang,
                method="keyword_partial",
                before_text_snippet=before_text[:100],
                after_text_snippet=after_text[:100],
            )

        before_template = BEFORE_TEMPLATES.get(lang, BEFORE_TEMPLATES["en"])
        after_template = AFTER_TEMPLATES.get(lang, AFTER_TEMPLATES["en"])

        sentences = [before_text, after_text, before_template, after_template]
        try:
            embeddings = model.encode(sentences, show_progress_bar=False)
        except Exception as exc:
            logger.warning(f"[nlp_pairing] Encoding failed: {exc}")
            return NLPPairingResult(
                pairing_score=0.0,
                detected_language=lang,
                method="error",
                before_text_snippet=before_text[:100],
                after_text_snippet=after_text[:100],
            )

        emb_before_text, emb_after_text, emb_before_tmpl, emb_after_tmpl = embeddings

        # Score: how well does each text match its expected template?
        sim_before = _cosine(emb_before_text, emb_before_tmpl)
        sim_after = _cosine(emb_after_text, emb_after_tmpl)

        # Cross-check: before-text should NOT match after-template better than before-template
        sim_before_cross = _cosine(emb_before_text, emb_after_tmpl)
        sim_after_cross = _cosine(emb_after_text, emb_before_tmpl)

        ordering_penalty = 0.0
        if sim_before_cross > sim_before:
            ordering_penalty += 0.2  # text labelled "before" matches after-template better
        if sim_after_cross > sim_after:
            ordering_penalty += 0.2

        pairing_score = max(0.0, ((sim_before + sim_after) / 2) - ordering_penalty)

        return NLPPairingResult(
            pairing_score=round(pairing_score, 4),
            detected_language=lang,
            method="semantic",
            before_text_snippet=before_text[:100],
            after_text_snippet=after_text[:100],
        )

    @staticmethod
    def extract_image_context(img_url: str, page_html: str) -> str:
        """
        Extract surrounding text for a given image URL from the page HTML.
        Returns up to 300 characters of relevant context text.
        """
        from bs4 import BeautifulSoup, Tag

        soup = BeautifulSoup(page_html, "lxml")
        filename = img_url.split("/")[-1]

        img_tag = None
        for img in soup.find_all("img"):
            src = img.get("src") or img.get("data-src") or ""
            if img_url in src or (filename and filename in src):
                img_tag = img
                break

        if img_tag is None:
            return ""

        parts: list[str] = []

        # alt text
        alt = img_tag.get("alt") or ""
        if alt:
            parts.append(alt)

        # aria-label
        label = img_tag.get("aria-label") or ""
        if label:
            parts.append(label)

        # figcaption
        fig = img_tag.find_parent("figure")
        if fig:
            cap = fig.find("figcaption")
            if cap:
                parts.append(cap.get_text(strip=True))

        # next sibling text
        for sib in img_tag.next_siblings:
            if isinstance(sib, Tag) and sib.name in ("p", "span", "div", "caption"):
                text = sib.get_text(strip=True)
                if text:
                    parts.append(text)
                    break

        # parent element heading or title
        for parent in img_tag.parents:
            if not isinstance(parent, Tag):
                continue
            heading = parent.find(re.compile(r"h[1-6]"))
            if heading:
                parts.append(heading.get_text(strip=True))
                break

        combined = " ".join(parts)
        return combined[:300]
