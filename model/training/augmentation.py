"""Data augmentation pipeline for before/after image pairs.

Augmentations are applied identically to both images in a pair so spatial
consistency is preserved — we never flip only the before image, for example.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from PIL import Image, ImageEnhance


@dataclass
class AugmentationConfig:
    horizontal_flip: bool = True
    brightness_jitter: float = 0.1
    contrast_jitter: float = 0.1
    saturation_jitter: float = 0.05
    hue_jitter: float = 0.02

    @classmethod
    def from_dict(cls, d: dict) -> "AugmentationConfig":
        return cls(
            horizontal_flip=d.get("horizontal_flip", True),
            brightness_jitter=d.get("brightness_jitter", 0.1),
            contrast_jitter=d.get("contrast_jitter", 0.1),
            saturation_jitter=d.get("saturation_jitter", 0.05),
            hue_jitter=d.get("hue_jitter", 0.02),
        )

    @classmethod
    def disabled(cls) -> "AugmentationConfig":
        """Return a config that applies no augmentations (for val/test splits)."""
        return cls(
            horizontal_flip=False,
            brightness_jitter=0.0,
            contrast_jitter=0.0,
            saturation_jitter=0.0,
            hue_jitter=0.0,
        )


class PairAugmentor:
    """Applies identical augmentations to a before/after image pair."""

    def __init__(self, config: AugmentationConfig) -> None:
        self.config = config

    def __call__(
        self, before: Image.Image, after: Image.Image
    ) -> tuple[Image.Image, Image.Image]:
        return self.augment(before, after)

    def augment(
        self, before: Image.Image, after: Image.Image
    ) -> tuple[Image.Image, Image.Image]:
        """Apply all configured augmentations. Same random state for both images."""
        if self.config.horizontal_flip and random.random() < 0.5:
            before = before.transpose(Image.FLIP_LEFT_RIGHT)
            after = after.transpose(Image.FLIP_LEFT_RIGHT)

        if self.config.brightness_jitter > 0:
            factor = 1.0 + random.uniform(
                -self.config.brightness_jitter, self.config.brightness_jitter
            )
            before = ImageEnhance.Brightness(before).enhance(factor)
            after = ImageEnhance.Brightness(after).enhance(factor)

        if self.config.contrast_jitter > 0:
            factor = 1.0 + random.uniform(
                -self.config.contrast_jitter, self.config.contrast_jitter
            )
            before = ImageEnhance.Contrast(before).enhance(factor)
            after = ImageEnhance.Contrast(after).enhance(factor)

        if self.config.saturation_jitter > 0:
            factor = 1.0 + random.uniform(
                -self.config.saturation_jitter, self.config.saturation_jitter
            )
            before = ImageEnhance.Color(before).enhance(factor)
            after = ImageEnhance.Color(after).enhance(factor)

        return before, after

    def is_noop(self) -> bool:
        """True if this augmentor makes no changes (all values disabled/zero)."""
        cfg = self.config
        return (
            not cfg.horizontal_flip
            and cfg.brightness_jitter == 0.0
            and cfg.contrast_jitter == 0.0
            and cfg.saturation_jitter == 0.0
        )
