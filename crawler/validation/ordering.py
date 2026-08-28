"""
Two-gate ordering validator.

Gate 1 — Explicit label (structural heuristics):
    Requires StructuralResult.has_explicit_label == True.
    If the source page does not carry an unambiguous before/after label, the
    pair is rejected at Gate 1 regardless of visual appearance.

Gate 2 — Treatment-presence check (visual heuristics):
    Available after Phase 2 adds MediaPipe landmark extraction.
    Currently implemented as a stub that always passes, and will be activated
    in Phase 2 once facial landmarks are available in the database.

    When active, Gate 2 checks:
      Botulinum Toxin — forehead line depth should be higher in 'before' image.
      HA Filler       — lip volume index should be lower in 'before' image.

OrderingResult carries confidence from both gates and a final verdict.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from crawler.validation.structural import StructuralResult


class OrderingVerdict(str, Enum):
    PASS = "pass"          # Both gates satisfied
    GATE1_FAIL = "gate1_fail"  # No explicit structural label
    GATE2_FAIL = "gate2_fail"  # Visual treatment-presence check failed
    BYPASS = "bypass"      # Gate 2 stub bypassed (Phase 2 not yet active)


@dataclass
class OrderingResult:
    """Combined ordering validation result."""

    verdict: OrderingVerdict
    gate1_passed: bool
    gate2_passed: bool        # True if Gate 2 active and passed; True if bypassed
    confidence: float         # 0.0–1.0 combined confidence
    gate1_signals: list[str]  # forwarded from StructuralResult
    gate2_notes: str          # diagnostic notes from Gate 2


class OrderingValidator:
    """
    Applies the two-gate ordering check to a candidate before/after pair.

    Gate 2 is a stub in Phase 1.  Call `activate_gate2(True)` when MediaPipe
    landmark data is available (Phase 2+).
    """

    def __init__(self, gate2_active: bool = False) -> None:
        self._gate2_active = gate2_active

    def activate_gate2(self, active: bool) -> None:
        self._gate2_active = active

    def validate(
        self,
        structural: StructuralResult,
        treatment_type: str = "",
        before_landmarks: dict | None = None,
        after_landmarks: dict | None = None,
    ) -> OrderingResult:
        """
        Run both ordering gates.

        Args:
            structural:       StructuralResult from Layer 1.
            treatment_type:   "botulinum_toxin" | "ha_filler" | ""
            before_landmarks: Landmark dict from MediaPipe (Phase 2+), or None.
            after_landmarks:  Landmark dict from MediaPipe (Phase 2+), or None.

        Returns:
            OrderingResult with verdict and per-gate details.
        """
        # ── Gate 1 ─────────────────────────────────────────────────────────────
        gate1_passed = structural.has_explicit_label
        if not gate1_passed:
            return OrderingResult(
                verdict=OrderingVerdict.GATE1_FAIL,
                gate1_passed=False,
                gate2_passed=False,
                confidence=0.0,
                gate1_signals=structural.before_label_signals + structural.after_label_signals,
                gate2_notes="Gate 1 failed — no explicit structural label.",
            )

        gate1_signals = structural.before_label_signals + structural.after_label_signals
        gate1_confidence = structural.confidence

        # ── Gate 2 ─────────────────────────────────────────────────────────────
        if not self._gate2_active:
            return OrderingResult(
                verdict=OrderingVerdict.PASS,
                gate1_passed=True,
                gate2_passed=True,
                confidence=gate1_confidence,
                gate1_signals=gate1_signals,
                gate2_notes="Gate 2 bypassed (Phase 1 — landmarks not yet available).",
            )

        gate2_passed, gate2_confidence, gate2_notes = self._run_gate2(
            treatment_type,
            before_landmarks or {},
            after_landmarks or {},
        )

        if not gate2_passed:
            return OrderingResult(
                verdict=OrderingVerdict.GATE2_FAIL,
                gate1_passed=True,
                gate2_passed=False,
                confidence=gate1_confidence * gate2_confidence,
                gate1_signals=gate1_signals,
                gate2_notes=gate2_notes,
            )

        combined_confidence = min(1.0, (gate1_confidence + gate2_confidence) / 2)
        return OrderingResult(
            verdict=OrderingVerdict.PASS,
            gate1_passed=True,
            gate2_passed=True,
            confidence=round(combined_confidence, 3),
            gate1_signals=gate1_signals,
            gate2_notes=gate2_notes,
        )

    # ── Gate 2 implementation (Phase 2 activation required) ──────────────────

    @staticmethod
    def _run_gate2(
        treatment_type: str,
        before_lm: dict,
        after_lm: dict,
    ) -> tuple[bool, float, str]:
        """
        Visual treatment-presence check.

        Returns (passed, confidence, diagnostic_notes).
        Currently, if landmark data is not populated, returns (True, 0.5, bypass).
        """
        if not before_lm or not after_lm:
            return True, 0.5, "Gate 2: landmark data absent — bypassed."

        if treatment_type == "botulinum_toxin":
            return _check_botulinum_ordering(before_lm, after_lm)
        elif treatment_type == "ha_filler":
            return _check_filler_ordering(before_lm, after_lm)
        else:
            # Unknown treatment — skip Gate 2
            return True, 0.5, f"Gate 2: unknown treatment '{treatment_type}' — bypassed."


def _check_botulinum_ordering(before_lm: dict, after_lm: dict) -> tuple[bool, float, str]:
    """
    Botulinum toxin ordering check:
    Forehead line depth should be greater in 'before' than 'after'.

    Landmark keys expected (set by Phase 2 pipeline):
      forehead_line_depth: float  (higher = deeper lines)
    """
    before_depth = before_lm.get("forehead_line_depth")
    after_depth = after_lm.get("forehead_line_depth")

    if before_depth is None or after_depth is None:
        return True, 0.5, "Gate 2 (botox): forehead_line_depth metric absent — bypassed."

    if before_depth >= after_depth:
        conf = min(1.0, 0.5 + abs(before_depth - after_depth))
        return True, conf, f"Gate 2 (botox): before_depth={before_depth:.3f} >= after_depth={after_depth:.3f}."
    else:
        return False, 0.3, (
            f"Gate 2 (botox) FAIL: before_depth={before_depth:.3f} < after_depth={after_depth:.3f}. "
            "Images may be reversed."
        )


def _check_filler_ordering(before_lm: dict, after_lm: dict) -> tuple[bool, float, str]:
    """
    HA Filler ordering check:
    Lip volume index should be lower in 'before' than 'after'.

    Landmark keys expected (set by Phase 2 pipeline):
      lip_volume_index: float  (higher = more volume)
    """
    before_vol = before_lm.get("lip_volume_index")
    after_vol = after_lm.get("lip_volume_index")

    if before_vol is None or after_vol is None:
        return True, 0.5, "Gate 2 (filler): lip_volume_index metric absent — bypassed."

    if before_vol <= after_vol:
        conf = min(1.0, 0.5 + abs(after_vol - before_vol))
        return True, conf, f"Gate 2 (filler): before_vol={before_vol:.3f} <= after_vol={after_vol:.3f}."
    else:
        return False, 0.3, (
            f"Gate 2 (filler) FAIL: before_vol={before_vol:.3f} > after_vol={after_vol:.3f}. "
            "Images may be reversed."
        )
