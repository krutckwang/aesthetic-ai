"""Tests for the training set query helper."""

from __future__ import annotations

from datetime import datetime

import pytest
from database.models import Quarantine, TreatmentLabel, ZoneLabel
from database.queries.training_set import query_training_set, TrainingRecord
from tests.database.conftest import make_image, make_pair


def test_query_returns_eligible_pair(db_session):
    b = make_image(db_session, "https://a.com/b.jpg", "/b.jpg")
    a = make_image(db_session, "https://a.com/a.jpg", "/a.jpg")
    make_pair(db_session, b, a, ordering_confidence="HIGH")
    records = query_training_set(db_session)
    assert len(records) == 1
    assert records[0].pair_id is not None


def test_query_excludes_quarantined_before_image(db_session):
    b = make_image(db_session, "https://b.com/b.jpg", "/b.jpg", consent_tier=3)
    a = make_image(db_session, "https://b.com/a.jpg", "/a.jpg")
    make_pair(db_session, b, a)
    q = Quarantine(image_id=b.id, reason="tier3", source_url=b.source_url,
                   assessed_at=datetime.utcnow())
    db_session.add(q)
    db_session.flush()
    records = query_training_set(db_session)
    assert len(records) == 0


def test_query_excludes_quarantined_after_image(db_session):
    b = make_image(db_session, "https://c.com/b.jpg", "/b.jpg")
    a = make_image(db_session, "https://c.com/a.jpg", "/a.jpg", consent_tier=3)
    make_pair(db_session, b, a)
    q = Quarantine(image_id=a.id, reason="tier3", source_url=a.source_url,
                   assessed_at=datetime.utcnow())
    db_session.add(q)
    db_session.flush()
    records = query_training_set(db_session)
    assert len(records) == 0


def test_query_excludes_low_ordering_by_default(db_session):
    b = make_image(db_session, "https://d.com/b.jpg", "/b.jpg")
    a = make_image(db_session, "https://d.com/a.jpg", "/a.jpg")
    make_pair(db_session, b, a, ordering_confidence="LOW")
    records = query_training_set(db_session)
    assert len(records) == 0


def test_query_includes_low_ordering_when_flag_set(db_session):
    b = make_image(db_session, "https://e.com/b.jpg", "/b.jpg")
    a = make_image(db_session, "https://e.com/a.jpg", "/a.jpg")
    make_pair(db_session, b, a, ordering_confidence="LOW")
    records = query_training_set(db_session, include_low_ordering=True)
    assert len(records) == 1


def test_query_returns_treatment_label(db_session):
    b = make_image(db_session, "https://f.com/b.jpg", "/b.jpg")
    a = make_image(db_session, "https://f.com/a.jpg", "/a.jpg")
    pair = make_pair(db_session, b, a)
    label = TreatmentLabel(pair_id=pair.id, treatment_category="botox",
                           treatment_brand="Botox", confidence=0.9)
    db_session.add(label)
    db_session.flush()
    records = query_training_set(db_session)
    assert records[0].treatment_category == "botox"
    assert records[0].treatment_brand == "Botox"


def test_query_returns_zone_codes(db_session):
    b = make_image(db_session, "https://g.com/b.jpg", "/b.jpg")
    a = make_image(db_session, "https://g.com/a.jpg", "/a.jpg")
    pair = make_pair(db_session, b, a)
    db_session.add(ZoneLabel(pair_id=pair.id, zone_code="lips", confidence=0.9))
    db_session.add(ZoneLabel(pair_id=pair.id, zone_code="forehead_lines", confidence=0.8))
    db_session.flush()
    records = query_training_set(db_session)
    assert "lips" in records[0].zone_codes
    assert "forehead_lines" in records[0].zone_codes


def test_training_record_has_paths(db_session):
    b = make_image(db_session, "https://h.com/b.jpg", "/images/b.jpg")
    a = make_image(db_session, "https://h.com/a.jpg", "/images/a.jpg")
    make_pair(db_session, b, a)
    records = query_training_set(db_session)
    assert records[0].before_path == "/images/b.jpg"
    assert records[0].after_path == "/images/a.jpg"
