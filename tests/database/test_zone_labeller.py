"""Tests for the zone labeller DB writer."""

from __future__ import annotations

from datetime import datetime

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from database.models import Base, Image, ImagePair, ZoneLabel
from database.labelling.zone_labeller import ZoneLabeller
from pipeline.segmentation.zone_mapper import ZoneMappingResult, ZonePresence


@pytest.fixture
def session():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine)
    s = factory()
    yield s
    s.close()


@pytest.fixture
def pair_id(session):
    b = Image(file_path="/b.jpg", source_url="https://x.com/b.jpg",
              domain="x.com", collected_at=datetime.utcnow(), consent_tier=1)
    a = Image(file_path="/a.jpg", source_url="https://x.com/a.jpg",
              domain="x.com", collected_at=datetime.utcnow(), consent_tier=1)
    session.add_all([b, a])
    session.flush()
    pair = ImagePair(before_image_id=b.id, after_image_id=a.id, created_at=datetime.utcnow())
    session.add(pair)
    session.flush()
    return pair.id


def make_zone_result(zones: list[tuple[str, float]]) -> ZoneMappingResult:
    return ZoneMappingResult(
        zones=[ZonePresence(zone_code=z, confidence=c, landmark_count=5,
                            centroid_x=0.5, centroid_y=0.5) for z, c in zones],
        success=True,
        total_zones_checked=10,
    )


def test_writes_labels_above_threshold(session, pair_id):
    labeller = ZoneLabeller(confidence_threshold=0.5)
    result = make_zone_result([("lips", 0.9), ("forehead_lines", 0.8)])
    count = labeller.write_labels(session, pair_id, result)
    session.flush()
    assert count == 2
    rows = session.query(ZoneLabel).filter_by(pair_id=pair_id).all()
    assert len(rows) == 2


def test_skips_zones_below_threshold(session, pair_id):
    labeller = ZoneLabeller(confidence_threshold=0.5)
    result = make_zone_result([("lips", 0.9), ("chin", 0.2)])
    count = labeller.write_labels(session, pair_id, result)
    assert count == 1


def test_returns_zero_on_failed_result(session, pair_id):
    labeller = ZoneLabeller()
    failed = ZoneMappingResult(zones=[], success=False, total_zones_checked=0)
    count = labeller.write_labels(session, pair_id, failed)
    assert count == 0


def test_zone_code_stored_correctly(session, pair_id):
    labeller = ZoneLabeller()
    result = make_zone_result([("nasolabial_folds", 0.75)])
    labeller.write_labels(session, pair_id, result)
    session.flush()
    row = session.query(ZoneLabel).filter_by(pair_id=pair_id).first()
    assert row.zone_code == "nasolabial_folds"


def test_source_is_auto(session, pair_id):
    labeller = ZoneLabeller()
    result = make_zone_result([("lips", 0.9)])
    labeller.write_labels(session, pair_id, result)
    session.flush()
    row = session.query(ZoneLabel).filter_by(pair_id=pair_id).first()
    assert row.source == "auto"


def test_confidence_stored(session, pair_id):
    labeller = ZoneLabeller()
    result = make_zone_result([("cheek_malar", 0.88)])
    labeller.write_labels(session, pair_id, result)
    session.flush()
    row = session.query(ZoneLabel).filter_by(pair_id=pair_id).first()
    assert abs(row.confidence - 0.88) < 0.01
