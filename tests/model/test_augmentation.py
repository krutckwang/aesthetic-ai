"""Tests for the pair augmentation pipeline."""

from __future__ import annotations

import pytest
from PIL import Image

from model.training.augmentation import AugmentationConfig, PairAugmentor


def _img(w=64, h=64) -> Image.Image:
    return Image.new("RGB", (w, h), color=(128, 100, 80))


def test_disabled_config_is_noop():
    aug = PairAugmentor(AugmentationConfig.disabled())
    assert aug.is_noop() is True


def test_default_config_is_not_noop():
    aug = PairAugmentor(AugmentationConfig())
    assert aug.is_noop() is False


def test_augment_returns_same_size():
    aug = PairAugmentor(AugmentationConfig.disabled())
    b, a = aug.augment(_img(64, 64), _img(64, 64))
    assert b.size == (64, 64)
    assert a.size == (64, 64)


def test_augment_preserves_mode():
    aug = PairAugmentor(AugmentationConfig.disabled())
    b, a = aug.augment(_img(), _img())
    assert b.mode == "RGB"
    assert a.mode == "RGB"


def test_from_dict_parses_correctly():
    cfg = AugmentationConfig.from_dict({
        "horizontal_flip": False,
        "brightness_jitter": 0.2,
        "contrast_jitter": 0.15,
    })
    assert cfg.horizontal_flip is False
    assert cfg.brightness_jitter == 0.2
    assert cfg.contrast_jitter == 0.15


def test_callable_interface():
    aug = PairAugmentor(AugmentationConfig.disabled())
    b, a = aug(_img(), _img())
    assert isinstance(b, Image.Image)
    assert isinstance(a, Image.Image)


def test_horizontal_flip_deterministic_with_seed():
    import random
    cfg = AugmentationConfig(horizontal_flip=True, brightness_jitter=0, contrast_jitter=0,
                              saturation_jitter=0, hue_jitter=0)
    aug = PairAugmentor(cfg)
    random.seed(0)
    b1, a1 = aug.augment(_img(), _img())
    random.seed(0)
    b2, a2 = aug.augment(_img(), _img())
    assert list(b1.getdata()) == list(b2.getdata())


def test_both_images_flipped_together():
    """If flip fires, both images must be flipped — they stay consistent."""
    import random
    cfg = AugmentationConfig(horizontal_flip=True, brightness_jitter=0, contrast_jitter=0,
                              saturation_jitter=0, hue_jitter=0)
    aug = PairAugmentor(cfg)
    # Asymmetric image so we can detect a flip
    before = Image.new("RGB", (10, 10))
    before.putpixel((0, 0), (255, 0, 0))   # red pixel top-left
    after = Image.new("RGB", (10, 10))
    after.putpixel((0, 0), (0, 255, 0))    # green pixel top-left

    random.seed(0)  # May or may not flip — run many times to hit both cases
    flipped_before = []
    flipped_after = []
    for _ in range(50):
        b, a = aug.augment(before.copy(), after.copy())
        flipped_before.append(b.getpixel((0, 0)) != (255, 0, 0))
        flipped_after.append(a.getpixel((0, 0)) != (0, 255, 0))

    # Whenever before was flipped, after was also flipped
    for bf, af in zip(flipped_before, flipped_after):
        assert bf == af, "Before and after images must flip in sync"
