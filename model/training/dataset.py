"""PyTorch Dataset for aesthetic before/after image pairs.

Supports two input modes:
  - manifest JSON (Kaggle use case): list of dicts with before_path/after_path/labels
  - TrainingRecord list (Oracle/local use case): output of query_training_set()

The dataset returns dicts ready for InstructPix2Pix training.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image
from torch.utils.data import Dataset

from model.training.augmentation import AugmentationConfig, PairAugmentor


# ── Instruction prompt templates ──────────────────────────────────────────────

_TREATMENT_PROMPTS: dict[str, str] = {
    "botox": "Apply botulinum toxin to smooth forehead lines and glabellar frown lines",
    "lip_filler": "Apply hyaluronic acid filler to enhance lip volume and definition",
    "cheek_filler": "Apply hyaluronic acid filler to add volume to the cheeks and malar area",
    "under_eye_filler": "Apply hyaluronic acid filler to the tear trough to reduce under-eye hollows",
    "jawline_filler": "Apply filler to define and contour the jawline",
    "dermal_filler": "Apply dermal filler to restore facial volume",
    "rhinoplasty": "Apply non-surgical rhinoplasty to reshape the nose",
    "blepharoplasty": "Apply eyelid treatment to refresh the periorbital area",
    "facelift": "Apply facial rejuvenation treatment to lift and tighten",
    "thread_lift": "Apply thread lift for non-surgical facial lifting",
    "laser_resurfacing": "Apply laser resurfacing to improve skin texture and tone",
    "chemical_peel": "Apply chemical peel to improve skin tone and reduce fine lines",
    "microneedling": "Apply microneedling to stimulate collagen and improve skin texture",
    "prp": "Apply PRP treatment for facial rejuvenation and skin quality",
    "kybella": "Apply Kybella to reduce submental fat under the chin",
}
_DEFAULT_PROMPT = "Apply aesthetic facial treatment"


@dataclass
class PairRecord:
    pair_id: int
    before_path: str
    after_path: str
    treatment_category: str | None = None
    treatment_brand: str | None = None
    zone_codes: list[str] | None = None


def build_instruction(record: PairRecord) -> str:
    """Build the text instruction for InstructPix2Pix from a pair's treatment label."""
    base = _TREATMENT_PROMPTS.get(record.treatment_category or "", _DEFAULT_PROMPT)
    if record.treatment_brand:
        base = f"{base} using {record.treatment_brand}"
    return base


def _load_record(d: dict) -> PairRecord:
    return PairRecord(
        pair_id=d.get("pair_id", 0),
        before_path=d["before_path"],
        after_path=d["after_path"],
        treatment_category=d.get("treatment_category"),
        treatment_brand=d.get("treatment_brand"),
        zone_codes=d.get("zone_codes", []),
    )


class AestheticPairDataset(Dataset):
    """
    Returns dicts with keys:
        pixel_values        — before image tensor (C, H, W), normalised [-1, 1]
        edited_pixel_values — after image tensor (C, H, W), normalised [-1, 1]
        input_ids           — tokenised instruction (if tokenizer provided)
        instruction         — raw instruction string (always present)
    """

    def __init__(
        self,
        records: list[PairRecord],
        image_size: int = 512,
        augmentation_config: AugmentationConfig | None = None,
        tokenizer=None,
        max_token_length: int = 77,
    ) -> None:
        self.records = records
        self.image_size = image_size
        self.augmentor = PairAugmentor(augmentation_config or AugmentationConfig.disabled())
        self.tokenizer = tokenizer
        self.max_token_length = max_token_length

    # ── Public ────────────────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        record = self.records[idx]
        before = self._load_image(record.before_path)
        after = self._load_image(record.after_path)
        before, after = self.augmentor.augment(before, after)

        instruction = build_instruction(record)
        item: dict[str, Any] = {
            "instruction": instruction,
            "pixel_values": self._to_tensor(before),
            "edited_pixel_values": self._to_tensor(after),
            "pair_id": record.pair_id,
        }
        if self.tokenizer is not None:
            item["input_ids"] = self._tokenize(instruction)
        return item

    # ── Class methods ─────────────────────────────────────────────────────────

    @classmethod
    def from_manifest(cls, manifest_path: str | Path, **kwargs) -> "AestheticPairDataset":
        """Load records from a DVC-exported JSON manifest file."""
        data = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
        records = [_load_record(d) for d in data]
        return cls(records, **kwargs)

    @classmethod
    def split(
        cls,
        records: list[PairRecord],
        train_frac: float = 0.80,
        val_frac: float = 0.10,
        seed: int = 42,
        **kwargs,
    ) -> tuple["AestheticPairDataset", "AestheticPairDataset", "AestheticPairDataset"]:
        """Split records into train/val/test datasets. Returns (train, val, test)."""
        rng = random.Random(seed)
        shuffled = list(records)
        rng.shuffle(shuffled)

        n = len(shuffled)
        n_train = int(n * train_frac)
        n_val = int(n * val_frac)

        train_records = shuffled[:n_train]
        val_records = shuffled[n_train: n_train + n_val]
        test_records = shuffled[n_train + n_val:]

        aug = kwargs.pop("augmentation_config", AugmentationConfig())
        train_ds = cls(train_records, augmentation_config=aug, **kwargs)
        val_ds = cls(val_records, augmentation_config=AugmentationConfig.disabled(), **kwargs)
        test_ds = cls(test_records, augmentation_config=AugmentationConfig.disabled(), **kwargs)
        return train_ds, val_ds, test_ds

    # ── Private ───────────────────────────────────────────────────────────────

    def _load_image(self, path: str) -> Image.Image:
        img = Image.open(path).convert("RGB")
        return img.resize((self.image_size, self.image_size), Image.LANCZOS)

    @staticmethod
    def _to_tensor(img: Image.Image):
        import torch
        import torchvision.transforms.functional as TF
        tensor = TF.to_tensor(img)        # [0, 1]
        return tensor * 2.0 - 1.0         # [-1, 1]

    def _tokenize(self, text: str):
        return self.tokenizer(
            text,
            max_length=self.max_token_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        ).input_ids.squeeze(0)
