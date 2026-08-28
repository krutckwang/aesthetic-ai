"""Tests for crawler/validation/ordering.py — two-gate ordering validator."""

from __future__ import annotations

import pytest

from crawler.validation.ordering import OrderingResult, OrderingValidator, OrderingVerdict
from crawler.validation.structural import StructuralResult


def _structural(has_explicit: bool, confidence: float = 0.85) -> StructuralResult:
    return StructuralResult(
        has_explicit_label=has_explicit,
        confidence=confidence,
        before_label_signals=["strong:alt_text:before"] if has_explicit else [],
        after_label_signals=["strong:alt_text:after"] if has_explicit else [],
        before_ordering="before" if has_explicit else "unknown",
        after_ordering="after" if has_explicit else "unknown",
    )


class TestGate1:
    def test_no_explicit_label_fails_gate1(self):
        validator = OrderingValidator()
        result = validator.validate(_structural(has_explicit=False))
        assert result.verdict == OrderingVerdict.GATE1_FAIL
        assert result.gate1_passed is False
        assert result.confidence == 0.0

    def test_explicit_label_passes_gate1(self):
        validator = OrderingValidator()
        result = validator.validate(_structural(has_explicit=True))
        assert result.gate1_passed is True
        assert result.verdict != OrderingVerdict.GATE1_FAIL

    def test_gate1_signals_forwarded_to_result(self):
        validator = OrderingValidator()
        result = validator.validate(_structural(has_explicit=True))
        assert len(result.gate1_signals) > 0

    def test_gate1_fail_confidence_is_zero(self):
        validator = OrderingValidator()
        result = validator.validate(_structural(has_explicit=False, confidence=0.9))
        assert result.confidence == 0.0


class TestGate2Bypass:
    def test_phase1_bypasses_gate2(self):
        validator = OrderingValidator(gate2_active=False)
        result = validator.validate(_structural(has_explicit=True))
        assert result.verdict == OrderingVerdict.PASS
        assert result.gate2_passed is True
        assert "bypass" in result.gate2_notes.lower()

    def test_activate_gate2_changes_behavior(self):
        validator = OrderingValidator(gate2_active=False)
        validator.activate_gate2(True)
        # With no landmarks provided, Gate 2 should still pass (no data = bypass)
        result = validator.validate(
            _structural(has_explicit=True),
            treatment_type="botulinum_toxin",
            before_landmarks=None,
            after_landmarks=None,
        )
        assert result.verdict == OrderingVerdict.PASS


class TestGate2BotoxOrdering:
    def test_correct_botox_ordering_passes(self):
        validator = OrderingValidator(gate2_active=True)
        result = validator.validate(
            _structural(has_explicit=True),
            treatment_type="botulinum_toxin",
            before_landmarks={"forehead_line_depth": 0.8},
            after_landmarks={"forehead_line_depth": 0.3},
        )
        assert result.verdict == OrderingVerdict.PASS
        assert result.gate2_passed is True

    def test_reversed_botox_ordering_fails(self):
        validator = OrderingValidator(gate2_active=True)
        result = validator.validate(
            _structural(has_explicit=True),
            treatment_type="botulinum_toxin",
            before_landmarks={"forehead_line_depth": 0.2},
            after_landmarks={"forehead_line_depth": 0.9},
        )
        assert result.verdict == OrderingVerdict.GATE2_FAIL
        assert result.gate2_passed is False

    def test_equal_botox_depth_passes(self):
        """Equal values are treated as non-reversed."""
        validator = OrderingValidator(gate2_active=True)
        result = validator.validate(
            _structural(has_explicit=True),
            treatment_type="botulinum_toxin",
            before_landmarks={"forehead_line_depth": 0.5},
            after_landmarks={"forehead_line_depth": 0.5},
        )
        assert result.verdict == OrderingVerdict.PASS


class TestGate2FillerOrdering:
    def test_correct_filler_ordering_passes(self):
        validator = OrderingValidator(gate2_active=True)
        result = validator.validate(
            _structural(has_explicit=True),
            treatment_type="ha_filler",
            before_landmarks={"lip_volume_index": 0.2},
            after_landmarks={"lip_volume_index": 0.8},
        )
        assert result.verdict == OrderingVerdict.PASS

    def test_reversed_filler_ordering_fails(self):
        validator = OrderingValidator(gate2_active=True)
        result = validator.validate(
            _structural(has_explicit=True),
            treatment_type="ha_filler",
            before_landmarks={"lip_volume_index": 0.9},
            after_landmarks={"lip_volume_index": 0.3},
        )
        assert result.verdict == OrderingVerdict.GATE2_FAIL


class TestUnknownTreatment:
    def test_unknown_treatment_bypasses_gate2(self):
        validator = OrderingValidator(gate2_active=True)
        result = validator.validate(
            _structural(has_explicit=True),
            treatment_type="laser_resurfacing",
            before_landmarks={"some_metric": 0.5},
            after_landmarks={"some_metric": 0.2},
        )
        # Unknown treatment should not fail Gate 2
        assert result.verdict == OrderingVerdict.PASS
