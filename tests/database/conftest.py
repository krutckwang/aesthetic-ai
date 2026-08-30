"""Shared fixtures for database tests."""

from __future__ import annotations

from datetime import datetime

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session

from database.models import (
    Base, Image, ImagePair, TreatmentLabel, ZoneLabel, Quarantine,
)
from database.session import override_engine


@pytest.fixture
def db_engine():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    override_engine(engine)
    return engine


@pytest.fixture
def db_session(db_engine) -> Session:
    factory = sessionmaker(bind=db_engine)
    session = factory()
    yield session
    session.close()


def make_image(session: Session, url: str, file_path: str = "/img.jpg",
               consent_tier: int = 1) -> Image:
    img = Image(
        file_path=file_path,
        source_url=url,
        domain=url.split("/")[2],
        collected_at=datetime.utcnow(),
        consent_tier=consent_tier,
    )
    session.add(img)
    session.flush()
    return img


def make_pair(session: Session, before: Image, after: Image,
              ordering_confidence: str = "HIGH") -> ImagePair:
    pair = ImagePair(
        before_image_id=before.id,
        after_image_id=after.id,
        ordering_confidence=ordering_confidence,
        created_at=datetime.utcnow(),
    )
    session.add(pair)
    session.flush()
    return pair
