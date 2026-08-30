"""Tests for the multilingual treatment labeller."""

from __future__ import annotations

import pytest
from database.labelling.treatment_labeller import TreatmentLabeller, extract_treatment


@pytest.fixture
def labeller():
    return TreatmentLabeller()


def test_botox_brand_detected(labeller):
    result = labeller.label("Patient received Botox injections to forehead")
    assert result.treatment_category == "botox"
    assert result.treatment_brand == "Botox"
    assert result.confidence >= 0.5


def test_dysport_detected(labeller):
    result = labeller.label("Dysport treatment for glabellar lines")
    assert result.treatment_category == "botox"
    assert result.treatment_brand == "Dysport"


def test_lip_filler_detected(labeller):
    result = labeller.label("Lip filler augmentation with Juvederm")
    assert result.treatment_category == "lip_filler"


def test_cheek_filler_detected(labeller):
    result = labeller.label("Cheek filler using Voluma for malar enhancement")
    assert result.treatment_category == "cheek_filler"
    assert result.treatment_brand == "Juvederm Voluma"


def test_rhinoplasty_detected(labeller):
    result = labeller.label("Non surgical rhinoplasty before and after results")
    assert result.treatment_category == "rhinoplasty"


def test_generic_filler_brand_detected(labeller):
    result = labeller.label("Restylane injection for nasolabial folds")
    assert result.treatment_category == "dermal_filler"
    assert result.treatment_brand == "Restylane"


def test_unknown_text_returns_none(labeller):
    result = labeller.label("Before and after photos of the patient")
    assert result.treatment_category is None
    assert result.treatment_brand is None
    assert result.confidence == 0.0


def test_empty_text_returns_none(labeller):
    result = labeller.label("")
    assert result.treatment_category is None
    assert result.confidence == 0.0


def test_korean_botox_detected(labeller):
    result = labeller.label("보톡스 시술 전후 사진")
    assert result.treatment_category == "botox"


def test_portuguese_lip_filler_detected(labeller):
    result = labeller.label("preenchimento labial com ácido hialurônico")
    assert result.treatment_category == "lip_filler"


def test_label_pair_combines_texts(labeller):
    # Neither text alone is definitive but together they are
    result = labeller.label_pair(["before and after", "lip filler augmentation"])
    assert result.treatment_category == "lip_filler"


def test_confidence_below_threshold_returns_none():
    labeller = TreatmentLabeller(confidence_threshold=0.99)
    result = labeller.label("Botox injection")
    # Single match → confidence = 0.4 + 0.3 = 0.7 < 0.99
    assert result.treatment_category is None


def test_matched_rule_populated_on_success(labeller):
    result = labeller.label("Juvederm Voluma cheek filler treatment")
    assert result.matched_rule is not None
