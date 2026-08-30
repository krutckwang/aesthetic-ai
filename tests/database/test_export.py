"""Tests for the DVC manifest export."""

from __future__ import annotations

import json

import pytest
from database.models import TreatmentLabel
from database.queries.export import export_manifest
from tests.database.conftest import make_image, make_pair


def test_export_creates_json_file(db_session, tmp_path):
    b = make_image(db_session, "https://x.com/b.jpg", "/b.jpg")
    a = make_image(db_session, "https://x.com/a.jpg", "/a.jpg")
    make_pair(db_session, b, a)
    out = tmp_path / "manifest.json"
    count = export_manifest(db_session, out)
    assert out.exists()
    assert count == 1


def test_export_returns_correct_count(db_session, tmp_path):
    for i in range(3):
        b = make_image(db_session, f"https://y.com/b{i}.jpg", f"/b{i}.jpg")
        a = make_image(db_session, f"https://y.com/a{i}.jpg", f"/a{i}.jpg")
        make_pair(db_session, b, a)
    out = tmp_path / "manifest.json"
    count = export_manifest(db_session, out)
    assert count == 3


def test_export_is_sorted_by_pair_id(db_session, tmp_path):
    for i in range(4):
        b = make_image(db_session, f"https://z.com/b{i}.jpg", f"/b{i}.jpg")
        a = make_image(db_session, f"https://z.com/a{i}.jpg", f"/a{i}.jpg")
        make_pair(db_session, b, a)
    out = tmp_path / "manifest.json"
    export_manifest(db_session, out)
    data = json.loads(out.read_text())
    ids = [r["pair_id"] for r in data]
    assert ids == sorted(ids)


def test_manifest_record_schema(db_session, tmp_path):
    b = make_image(db_session, "https://s.com/b.jpg", "/b.jpg")
    a = make_image(db_session, "https://s.com/a.jpg", "/a.jpg")
    pair = make_pair(db_session, b, a)
    db_session.add(TreatmentLabel(pair_id=pair.id, treatment_category="botox", confidence=0.9))
    db_session.flush()
    out = tmp_path / "manifest.json"
    export_manifest(db_session, out)
    data = json.loads(out.read_text())
    record = data[0]
    assert "pair_id" in record
    assert "before_path" in record
    assert "after_path" in record
    assert "treatment_category" in record
    assert "treatment_brand" in record
    assert "zone_codes" in record
    assert record["treatment_category"] == "botox"


def test_export_creates_parent_directories(db_session, tmp_path):
    b = make_image(db_session, "https://t.com/b.jpg", "/b.jpg")
    a = make_image(db_session, "https://t.com/a.jpg", "/a.jpg")
    make_pair(db_session, b, a)
    nested = tmp_path / "deep" / "nested" / "manifest.json"
    export_manifest(db_session, nested)
    assert nested.exists()
