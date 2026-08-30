"""Tests for the class balance reporter."""

from __future__ import annotations

import pytest
from database.models import TreatmentLabel
from database.queries.balance import compute_balance
from tests.database.conftest import make_image, make_pair


def _add_pair_with_label(session, url_prefix: str, i: int, category: str | None):
    b = make_image(session, f"https://{url_prefix}.com/b{i}.jpg", f"/{url_prefix}_b{i}.jpg")
    a = make_image(session, f"https://{url_prefix}.com/a{i}.jpg", f"/{url_prefix}_a{i}.jpg")
    pair = make_pair(session, b, a)
    if category:
        session.add(TreatmentLabel(pair_id=pair.id, treatment_category=category, confidence=0.9))
        session.flush()
    return pair


def test_balance_counts_by_category(db_session):
    _add_pair_with_label(db_session, "p1", 0, "botox")
    _add_pair_with_label(db_session, "p1", 1, "botox")
    _add_pair_with_label(db_session, "p1", 2, "lip_filler")
    report = compute_balance(db_session)
    assert report.counts["botox"] == 2
    assert report.counts["lip_filler"] == 1


def test_balance_counts_unlabelled(db_session):
    _add_pair_with_label(db_session, "p2", 0, "botox")
    _add_pair_with_label(db_session, "p2", 1, None)
    report = compute_balance(db_session)
    assert report.unlabelled == 1


def test_balance_total_includes_unlabelled(db_session):
    _add_pair_with_label(db_session, "p3", 0, "botox")
    _add_pair_with_label(db_session, "p3", 1, None)
    report = compute_balance(db_session)
    assert report.total == 2


def test_balance_weights_inverse_frequency(db_session):
    # 4 botox, 2 lip_filler → total=6, weight botox=6/4=1.5, lip_filler=6/2=3.0
    for i in range(4):
        _add_pair_with_label(db_session, "p4", i, "botox")
    for i in range(2):
        _add_pair_with_label(db_session, "p4", 10 + i, "lip_filler")
    report = compute_balance(db_session)
    assert abs(report.weights["botox"] - 1.5) < 0.01
    assert abs(report.weights["lip_filler"] - 3.0) < 0.01


def test_balance_empty_db(db_session):
    report = compute_balance(db_session)
    assert report.total == 0
    assert report.unlabelled == 0
    assert report.counts == {}
    assert report.weights == {}


def test_balance_minority_class_gets_higher_weight(db_session):
    for i in range(10):
        _add_pair_with_label(db_session, "p5", i, "botox")
    _add_pair_with_label(db_session, "p5", 99, "rhinoplasty")
    report = compute_balance(db_session)
    assert report.weights["rhinoplasty"] > report.weights["botox"]
