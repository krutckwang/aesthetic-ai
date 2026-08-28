"""Shared pytest fixtures for all test modules."""

from __future__ import annotations

from pathlib import Path

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from crawler.base import ConsentTier, RawImagePair
from database.models import Base
from database.session import override_engine


@pytest.fixture(scope="function")
def tmp_db(tmp_path: Path):
    """In-memory SQLite engine with all tables created. Torn down after each test."""
    db_path = tmp_path / "test.db"
    engine = create_engine(f"sqlite:///{db_path}", connect_args={"check_same_thread": False})
    Base.metadata.create_all(engine)
    override_engine(engine)
    yield engine
    Base.metadata.drop_all(engine)


@pytest.fixture
def tmp_storage(tmp_path: Path) -> Path:
    """Temporary directory simulating Oracle block storage."""
    storage = tmp_path / "storage"
    storage.mkdir()
    return storage


@pytest.fixture
def tmp_queue_db(tmp_path: Path) -> Path:
    """Path to a temporary staging queue SQLite file."""
    return tmp_path / "queue.db"


@pytest.fixture
def sample_pair() -> RawImagePair:
    """A valid Tier 1 RawImagePair for use in tests."""
    return RawImagePair(
        before_url="https://www.realself.com/images/before_001.jpg",
        after_url="https://www.realself.com/images/after_001.jpg",
        source_url="https://www.realself.com/botox/reviews/1234",
        source_name="realself",
        language="en",
        consent_tier=ConsentTier.CONFIRMED,
        metadata={"treatment_name": "Botox", "provider_name": "Dr. Smith"},
    )


@pytest.fixture
def sample_pair_tier3() -> RawImagePair:
    """A Tier 3 (uncertain consent) pair for quarantine tests."""
    return RawImagePair(
        before_url="https://www.instagram.com/p/abc/before.jpg",
        after_url="https://www.instagram.com/p/abc/after.jpg",
        source_url="https://www.instagram.com/p/abc/",
        source_name="instagram",
        language="en",
        consent_tier=ConsentTier.UNCERTAIN,
        metadata={},
    )
