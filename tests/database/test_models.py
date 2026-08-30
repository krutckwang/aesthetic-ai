"""Tests for SQLAlchemy ORM models — CRUD and quarantine isolation."""

from __future__ import annotations

from datetime import datetime

import pytest
from sqlalchemy import create_engine, inspect
from sqlalchemy.orm import sessionmaker

from database.models import (
    Base, Image, ImagePair, TreatmentLabel, ZoneLabel,
    QualityScore, Landmark, ConsentRecord, Quarantine, SourceMetadata,
)
from database.session import override_engine


@pytest.fixture
def engine():
    eng = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(eng)
    return eng


@pytest.fixture
def session(engine):
    factory = sessionmaker(bind=engine)
    override_engine(engine)
    s = factory()
    yield s
    s.close()


def _image(source_url: str = "https://example.com/img.jpg", consent_tier: int = 1) -> Image:
    return Image(
        file_path="/images/img.jpg",
        source_url=source_url,
        domain="example.com",
        collected_at=datetime.utcnow(),
        consent_tier=consent_tier,
    )


def test_image_crud(session):
    img = _image()
    session.add(img)
    session.flush()
    assert img.id is not None
    fetched = session.get(Image, img.id)
    assert fetched.domain == "example.com"


def test_image_source_url_unique(session):
    session.add(_image("https://dup.com/a.jpg"))
    session.flush()
    session.add(_image("https://dup.com/a.jpg"))
    with pytest.raises(Exception):
        session.flush()


def test_image_pair_crud(session):
    b = _image("https://x.com/b.jpg")
    a = _image("https://x.com/a.jpg")
    session.add_all([b, a])
    session.flush()
    pair = ImagePair(
        before_image_id=b.id, after_image_id=a.id,
        ordering_confidence="HIGH", created_at=datetime.utcnow(),
    )
    session.add(pair)
    session.flush()
    assert pair.id is not None


def test_treatment_label_crud(session):
    b = _image("https://x.com/c.jpg")
    a = _image("https://x.com/d.jpg")
    session.add_all([b, a])
    session.flush()
    pair = ImagePair(before_image_id=b.id, after_image_id=a.id, created_at=datetime.utcnow())
    session.add(pair)
    session.flush()
    label = TreatmentLabel(pair_id=pair.id, treatment_category="botox", confidence=0.9)
    session.add(label)
    session.flush()
    assert label.id is not None
    assert label.treatment_category == "botox"


def test_zone_label_crud(session):
    b = _image("https://x.com/e.jpg")
    a = _image("https://x.com/f.jpg")
    session.add_all([b, a])
    session.flush()
    pair = ImagePair(before_image_id=b.id, after_image_id=a.id, created_at=datetime.utcnow())
    session.add(pair)
    session.flush()
    zl = ZoneLabel(pair_id=pair.id, zone_code="lips", confidence=0.85)
    session.add(zl)
    session.flush()
    assert zl.id is not None


def test_quality_score_crud(session):
    img = _image("https://x.com/q.jpg")
    session.add(img)
    session.flush()
    qs = QualityScore(image_id=img.id, blur_score=200.0, overall_grade="PASS")
    session.add(qs)
    session.flush()
    assert qs.id is not None


def test_landmark_crud(session):
    img = _image("https://x.com/lm.jpg")
    session.add(img)
    session.flush()
    lm = Landmark(image_id=img.id, landmark_index=0, x=0.5, y=0.5, z=0.0)
    session.add(lm)
    session.flush()
    assert lm.id is not None


def test_consent_record_crud(session):
    img = _image("https://x.com/cr.jpg")
    session.add(img)
    session.flush()
    cr = ConsentRecord(image_id=img.id, consent_tier=1, assessed_at=datetime.utcnow())
    session.add(cr)
    session.flush()
    assert cr.id is not None


def test_quarantine_crud(session):
    img = _image("https://x.com/qr.jpg", consent_tier=3)
    session.add(img)
    session.flush()
    q = Quarantine(image_id=img.id, reason="uncertain_consent",
                   source_url=img.source_url, assessed_at=datetime.utcnow())
    session.add(q)
    session.flush()
    assert q.id is not None


def test_source_metadata_crud(session):
    b = _image("https://x.com/sm_b.jpg")
    a = _image("https://x.com/sm_a.jpg")
    session.add_all([b, a])
    session.flush()
    pair = ImagePair(before_image_id=b.id, after_image_id=a.id, created_at=datetime.utcnow())
    session.add(pair)
    session.flush()
    sm = SourceMetadata(pair_id=pair.id, source_name="realself", language="en")
    session.add(sm)
    session.flush()
    assert sm.id is not None


def test_quarantine_has_no_training_relationship(session):
    """Quarantine table has no relationship to ImagePair — verified by absence of FK."""
    insp = inspect(session.bind)
    fks = {fk["referred_table"] for fk in insp.get_foreign_keys("quarantine")}
    assert "image_pair" not in fks
