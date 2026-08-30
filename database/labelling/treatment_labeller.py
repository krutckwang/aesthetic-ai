"""Extract treatment category and brand from free-text using multilingual keyword matching."""

from __future__ import annotations

import re
from dataclasses import dataclass

CONFIDENCE_THRESHOLD = 0.5

# Higher number = more specific category (wins over lower-priority generics)
_CATEGORY_PRIORITY: dict[str, int] = {
    "lip_filler": 3,
    "cheek_filler": 3,
    "under_eye_filler": 3,
    "jawline_filler": 3,
    "rhinoplasty": 2,
    "blepharoplasty": 2,
    "facelift": 2,
    "thread_lift": 2,
    "laser_resurfacing": 2,
    "chemical_peel": 2,
    "microneedling": 2,
    "prp": 2,
    "kybella": 2,
    "botox": 2,
    "dermal_filler": 1,  # generic fallback
}

# Each rule: (compiled pattern, treatment_category, brand | None)
# Brand-level rules appear before generic category rules so the most specific match wins.
_RULES: list[tuple[re.Pattern, str, str | None]] = [
    # ── Neurotoxins ──────────────────────────────────────────────────────────
    (re.compile(r'\b(botox|btx)\b', re.I), 'botox', 'Botox'),
    (re.compile(r'\bdysport\b', re.I), 'botox', 'Dysport'),
    (re.compile(r'\bxeomin\b', re.I), 'botox', 'Xeomin'),
    (re.compile(r'\b(daxxify|revance)\b', re.I), 'botox', 'Daxxify'),
    (re.compile(r'\b(neurotoxin|neuromodulator|wrinkle[\s\-]?relaxer)\b', re.I), 'botox', None),
    # ── Lip filler ───────────────────────────────────────────────────────────
    (re.compile(r'\bvolbella\b', re.I), 'lip_filler', 'Juvederm Volbella'),
    (re.compile(r'\b(lip[\s\-]?filler|lip[\s\-]?augment|lip[\s\-]?inject|lip[\s\-]?plump)\b', re.I), 'lip_filler', None),
    # ── Cheek filler ─────────────────────────────────────────────────────────
    (re.compile(r'\bvoluma\b', re.I), 'cheek_filler', 'Juvederm Voluma'),
    (re.compile(r'\b(cheek[\s\-]?filler|cheek[\s\-]?augment|malar[\s\-]?filler)\b', re.I), 'cheek_filler', None),
    # ── Under-eye / tear trough ───────────────────────────────────────────────
    (re.compile(r'\b(tear[\s\-]?trough|under[\s\-]?eye[\s\-]?filler|restylane[\s\-]?eyelight)\b', re.I), 'under_eye_filler', None),
    # ── Jawline / chin filler ────────────────────────────────────────────────
    (re.compile(r'\b(jawline[\s\-]?filler|jaw[\s\-]?filler|chin[\s\-]?filler)\b', re.I), 'jawline_filler', None),
    # ── Generic filler brands ────────────────────────────────────────────────
    (re.compile(r'\bjuvederm\b', re.I), 'dermal_filler', 'Juvederm'),
    (re.compile(r'\brestylane\b', re.I), 'dermal_filler', 'Restylane'),
    (re.compile(r'\bsculptra\b', re.I), 'dermal_filler', 'Sculptra'),
    (re.compile(r'\bradiesse\b', re.I), 'dermal_filler', 'Radiesse'),
    (re.compile(r'\bbelotero\b', re.I), 'dermal_filler', 'Belotero'),
    (re.compile(r'\b(rha[\s\-]?collection|revanesse)\b', re.I), 'dermal_filler', None),
    (re.compile(r'\bdermal[\s\-]?filler\b', re.I), 'dermal_filler', None),
    # ── Rhinoplasty ──────────────────────────────────────────────────────────
    (re.compile(r'\b(rhinoplasty|nose[\s\-]?job|nose[\s\-]?reshap|rhinoplast)\b', re.I), 'rhinoplasty', None),
    (re.compile(r'\b(liquid[\s\-]?nose|non[\s\-]?surgical[\s\-]?nose|non[\s\-]?surgical[\s\-]?rhinoplasty)\b', re.I), 'rhinoplasty', None),
    # ── Blepharoplasty ───────────────────────────────────────────────────────
    (re.compile(r'\b(blepharoplasty|eyelid[\s\-]?surgery|eye[\s\-]?lift)\b', re.I), 'blepharoplasty', None),
    # ── Facelift / thread lift ────────────────────────────────────────────────
    (re.compile(r'\b(facelift|face[\s\-]?lift|rhytidectomy|mini[\s\-]?lift)\b', re.I), 'facelift', None),
    (re.compile(r'\bthread[\s\-]?lift\b', re.I), 'thread_lift', None),
    # ── Laser / resurfacing ──────────────────────────────────────────────────
    (re.compile(r'\b(laser[\s\-]?resurfac|co2[\s\-]?laser|fraxel|clear[\s\-]?lift|halo[\s\-]?laser)\b', re.I), 'laser_resurfacing', None),
    (re.compile(r'\b(chemical[\s\-]?peel|glycolic[\s\-]?peel|vi[\s\-]?peel|tca[\s\-]?peel)\b', re.I), 'chemical_peel', None),
    (re.compile(r'\bmicroneedl\w*\b', re.I), 'microneedling', None),
    (re.compile(r'\b(prp|platelet[\s\-]?rich[\s\-]?plasma)\b', re.I), 'prp', None),
    (re.compile(r'\b(kybella|deoxycholic)\b', re.I), 'kybella', None),
    # ── Multilingual — Korean ────────────────────────────────────────────────
    (re.compile(r'보톡스'), 'botox', 'Botox'),
    (re.compile(r'립\s*필러'), 'lip_filler', None),
    (re.compile(r'필러'), 'dermal_filler', None),
    # ── Multilingual — Portuguese ─────────────────────────────────────────────
    (re.compile(r'\b(preenchimento[\s\-]?labial|preenchimento[\s\-]?l[aá]bio)\b', re.I), 'lip_filler', None),
    (re.compile(r'\b(toxina[\s\-]?botul[ií]nica|aplica[cç][aã]o[\s\-]?botox)\b', re.I), 'botox', None),
    (re.compile(r'\bpreenchimento\b', re.I), 'dermal_filler', None),
]


@dataclass
class LabelResult:
    treatment_category: str | None
    treatment_brand: str | None
    confidence: float
    matched_rule: str | None


def extract_treatment(text: str, confidence_threshold: float = CONFIDENCE_THRESHOLD) -> LabelResult:
    """Return treatment category and brand from text. Returns None fields if uncertain."""
    if not text or not text.strip():
        return LabelResult(None, None, 0.0, None)

    matches: list[tuple[str, str | None, re.Pattern]] = []
    for pattern, category, brand in _RULES:
        if pattern.search(text):
            matches.append((category, brand, pattern))

    if not matches:
        return LabelResult(None, None, 0.0, None)

    # Confidence: base 0.4 + 0.4 per additional match, capped at 1.0
    confidence = min(0.4 + len(matches) * 0.3, 1.0)

    # Most specific match: prefer higher category priority, then brand presence as tiebreaker
    best_category, best_brand, best_pattern = max(
        matches,
        key=lambda m: (_CATEGORY_PRIORITY.get(m[0], 0), m[1] is not None),
    )

    # If a brand was matched for a generic category but a more specific category also matched,
    # carry the brand from the generic match onto the specific category.
    if best_brand is None:
        for cat, brand, _ in matches:
            if brand is not None and _CATEGORY_PRIORITY.get(cat, 0) <= _CATEGORY_PRIORITY.get(best_category, 0):
                best_brand = brand
                break

    if confidence < confidence_threshold:
        return LabelResult(None, None, round(confidence, 3), None)

    return LabelResult(
        treatment_category=best_category,
        treatment_brand=best_brand,
        confidence=round(confidence, 3),
        matched_rule=best_pattern.pattern,
    )


class TreatmentLabeller:
    """Wraps extract_treatment with a configurable threshold."""

    def __init__(self, confidence_threshold: float = CONFIDENCE_THRESHOLD):
        self._threshold = confidence_threshold

    def label(self, text: str) -> LabelResult:
        return extract_treatment(text, self._threshold)

    def label_pair(self, texts: list[str]) -> LabelResult:
        """Label from multiple text sources — combines all into one classification."""
        combined = " ".join(t for t in texts if t)
        return extract_treatment(combined, self._threshold)
