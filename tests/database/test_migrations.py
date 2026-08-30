"""Tests that the database schema is consistent and all expected tables are created."""

from __future__ import annotations

import pytest
from sqlalchemy import create_engine, inspect

from database.models import Base
from database.session import init_db


EXPECTED_TABLES = {
    "image",
    "image_pair",
    "treatment_label",
    "zone_label",
    "quality_score",
    "landmark",
    "consent_record",
    "quarantine",
    "source_metadata",
}


@pytest.fixture
def engine():
    eng = create_engine("sqlite:///:memory:")
    return eng


def test_init_db_creates_all_tables(engine):
    init_db(engine)
    insp = inspect(engine)
    tables = set(insp.get_table_names())
    assert EXPECTED_TABLES.issubset(tables)


def test_all_expected_tables_present(engine):
    Base.metadata.create_all(engine)
    insp = inspect(engine)
    tables = set(insp.get_table_names())
    for t in EXPECTED_TABLES:
        assert t in tables, f"Missing table: {t}"


def test_quarantine_table_has_no_pair_fk(engine):
    Base.metadata.create_all(engine)
    insp = inspect(engine)
    fks = {fk["referred_table"] for fk in insp.get_foreign_keys("quarantine")}
    assert "image_pair" not in fks


def test_image_pair_has_unique_constraint(engine):
    Base.metadata.create_all(engine)
    insp = inspect(engine)
    unique_constraints = insp.get_unique_constraints("image_pair")
    names = [uc["name"] for uc in unique_constraints]
    assert "uq_pair" in names


def test_landmark_has_unique_constraint(engine):
    Base.metadata.create_all(engine)
    insp = inspect(engine)
    unique_constraints = insp.get_unique_constraints("landmark")
    names = [uc["name"] for uc in unique_constraints]
    assert "uq_landmark" in names
