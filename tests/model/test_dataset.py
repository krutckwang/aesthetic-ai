"""Tests for the AestheticPairDataset."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# ── Mock torch and torchvision before any model imports ──────────────────────
torch_mock = MagicMock()
torch_utils_mock = MagicMock()
torch_utils_data_mock = MagicMock()
torch_utils_data_mock.Dataset = object  # Dataset base class → plain object
torch_utils_mock.data = torch_utils_data_mock
torch_mock.utils = torch_utils_mock

tv_mock = MagicMock()
tv_transforms_mock = MagicMock()
tv_transforms_functional_mock = MagicMock()

for mod, obj in [
    ("torch", torch_mock),
    ("torch.utils", torch_utils_mock),
    ("torch.utils.data", torch_utils_data_mock),
    ("torchvision", tv_mock),
    ("torchvision.transforms", tv_transforms_mock),
    ("torchvision.transforms.functional", tv_transforms_functional_mock),
]:
    sys.modules.setdefault(mod, obj)

from model.training.dataset import (  # noqa: E402
    AestheticPairDataset, PairRecord, build_instruction, _TREATMENT_PROMPTS,
)
from model.training.augmentation import AugmentationConfig  # noqa: E402
from PIL import Image  # noqa: E402
import numpy as np  # noqa: E402


def _record(
    pair_id: int = 1,
    treatment: str | None = "botox",
    brand: str | None = None,
    before: str = "/b.jpg",
    after: str = "/a.jpg",
) -> PairRecord:
    return PairRecord(
        pair_id=pair_id,
        before_path=before,
        after_path=after,
        treatment_category=treatment,
        treatment_brand=brand,
    )


def _write_dummy_images(tmp_path, pair_id=1):
    img = Image.new("RGB", (64, 64), color=(128, 100, 80))
    b = tmp_path / f"{pair_id}_b.jpg"
    a = tmp_path / f"{pair_id}_a.jpg"
    img.save(str(b))
    img.save(str(a))
    return str(b), str(a)


# ── build_instruction ─────────────────────────────────────────────────────────

def test_known_treatment_returns_specific_prompt():
    r = _record(treatment="botox")
    assert "botulinum" in build_instruction(r).lower()


def test_unknown_treatment_returns_default():
    r = _record(treatment="unknown_xyz")
    assert build_instruction(r) == "Apply aesthetic facial treatment"


def test_none_treatment_returns_default():
    r = _record(treatment=None)
    assert build_instruction(r) == "Apply aesthetic facial treatment"


def test_brand_appended_to_instruction():
    r = _record(treatment="lip_filler", brand="Juvederm Volbella")
    instruction = build_instruction(r)
    assert "Juvederm Volbella" in instruction


def test_all_treatment_keys_have_prompts():
    for key in _TREATMENT_PROMPTS:
        r = _record(treatment=key)
        prompt = build_instruction(r)
        assert len(prompt) > 10


# ── AestheticPairDataset ──────────────────────────────────────────────────────

def test_len(tmp_path):
    records = []
    for i in range(5):
        b, a = _write_dummy_images(tmp_path, i)
        records.append(_record(pair_id=i, before=b, after=a))
    ds = AestheticPairDataset(records, image_size=64)
    assert len(ds) == 5


def test_getitem_has_required_keys(tmp_path):
    b, a = _write_dummy_images(tmp_path)
    r = _record(before=b, after=a)
    ds = AestheticPairDataset([r], image_size=64)

    import torchvision.transforms.functional as TF
    TF.to_tensor = lambda img: MagicMock()

    item = ds[0]
    assert "instruction" in item
    assert "pixel_values" in item
    assert "edited_pixel_values" in item
    assert "pair_id" in item


def test_getitem_instruction_is_string(tmp_path):
    b, a = _write_dummy_images(tmp_path)
    ds = AestheticPairDataset([_record(before=b, after=a)], image_size=64)
    item = ds[0]
    assert isinstance(item["instruction"], str)


def test_split_sizes():
    records = [_record(pair_id=i) for i in range(100)]
    train_ds, val_ds, test_ds = AestheticPairDataset.split(
        records, train_frac=0.8, val_frac=0.1
    )
    assert len(train_ds) == 80
    assert len(val_ds) == 10
    assert len(test_ds) == 10


def test_split_total_equals_input():
    records = [_record(pair_id=i) for i in range(87)]
    train_ds, val_ds, test_ds = AestheticPairDataset.split(records)
    assert len(train_ds) + len(val_ds) + len(test_ds) == 87


def test_split_no_overlap():
    records = [_record(pair_id=i) for i in range(30)]
    train_ds, val_ds, test_ds = AestheticPairDataset.split(records)
    train_ids = {r.pair_id for r in train_ds.records}
    val_ids = {r.pair_id for r in val_ds.records}
    test_ids = {r.pair_id for r in test_ds.records}
    assert train_ids.isdisjoint(val_ids)
    assert train_ids.isdisjoint(test_ids)
    assert val_ids.isdisjoint(test_ids)


def test_split_is_deterministic():
    records = [_record(pair_id=i) for i in range(50)]
    train1, _, _ = AestheticPairDataset.split(records, seed=42)
    train2, _, _ = AestheticPairDataset.split(records, seed=42)
    assert [r.pair_id for r in train1.records] == [r.pair_id for r in train2.records]


def test_val_augmentation_disabled():
    records = [_record(pair_id=i) for i in range(10)]
    _, val_ds, test_ds = AestheticPairDataset.split(records)
    assert val_ds.augmentor.is_noop() is True
    assert test_ds.augmentor.is_noop() is True


def test_from_manifest(tmp_path):
    b, a = _write_dummy_images(tmp_path)
    manifest = [
        {"pair_id": 1, "before_path": b, "after_path": a,
         "treatment_category": "botox", "treatment_brand": None, "zone_codes": []}
    ]
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    ds = AestheticPairDataset.from_manifest(manifest_path, image_size=64)
    assert len(ds) == 1
