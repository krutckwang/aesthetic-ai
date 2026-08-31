"""Kaggle/Oracle training entry point.

Usage (Kaggle T4 notebook):

    !python model/training/train.py \
        --manifest /kaggle/input/aesthetic-pairs/manifest.json \
        --base_model timbrooks/instruct-pix2pix \
        --output_dir /kaggle/working/lora \
        --num_steps 15000 \
        --batch_size 4

Requires:
    - diffusers >= 0.25
    - peft
    - accelerate
    - transformers
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fine-tune InstructPix2Pix with LoRA")
    p.add_argument("--manifest", required=True, help="Path to DVC manifest JSON")
    p.add_argument("--base_model", default="timbrooks/instruct-pix2pix")
    p.add_argument("--output_dir", default="outputs/lora")
    p.add_argument("--num_steps", type=int, default=15_000)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--learning_rate", type=float, default=1e-4)
    p.add_argument("--mixed_precision", default="fp16", choices=["fp16", "bf16", "no"])
    p.add_argument("--lora_rank", type=int, default=16)
    p.add_argument("--lora_alpha", type=float, default=16.0)
    p.add_argument("--image_size", type=int, default=512)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--save_every", type=int, default=500)
    p.add_argument("--num_workers", type=int, default=2)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    import torch
    from torch.utils.data import DataLoader
    from diffusers import (
        StableDiffusionInstructPix2PixPipeline,
        DDPMScheduler,
    )
    from transformers import CLIPTokenizer

    from model.training.dataset import AestheticPairDataset
    from model.training.augmentation import AugmentationConfig
    from model.training.trainer import InstructPix2PixTrainer, TrainerConfig
    from model.instruct_pix2pix.lora import LoRAConfig, inject_lora, count_trainable_params

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Training on device: %s", device)

    # ── Load base pipeline components ─────────────────────────────────────────
    logger.info("Loading base model: %s", args.base_model)
    pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
        args.base_model,
        torch_dtype=torch.float16 if args.mixed_precision != "no" else torch.float32,
        safety_checker=None,
    )
    unet = pipe.unet
    vae = pipe.vae.to(device)
    text_encoder = pipe.text_encoder.to(device)
    tokenizer = pipe.tokenizer
    noise_scheduler = DDPMScheduler.from_pretrained(args.base_model, subfolder="scheduler")

    # ── Inject LoRA ───────────────────────────────────────────────────────────
    lora_config = LoRAConfig(rank=args.lora_rank, alpha=args.lora_alpha)
    unet = inject_lora(unet.to(device), lora_config)
    stats = count_trainable_params(unet)
    logger.info(
        "LoRA injected: %s trainable / %s total params (%.2f%%)",
        f"{stats['trainable']:,}",
        f"{stats['total']:,}",
        100.0 * stats["trainable"] / stats["total"],
    )

    # ── Dataset + DataLoaders ─────────────────────────────────────────────────
    aug_config = AugmentationConfig()
    train_ds, val_ds, _ = AestheticPairDataset.split(
        AestheticPairDataset.from_manifest(args.manifest, image_size=args.image_size).records,
        augmentation_config=aug_config,
        tokenizer=tokenizer,
    )
    logger.info("Train: %d  Val: %d", len(train_ds), len(val_ds))

    train_dl = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device == "cuda",
    )
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, num_workers=0) if len(val_ds) > 0 else None

    # ── Train ─────────────────────────────────────────────────────────────────
    trainer_config = TrainerConfig(
        output_dir=args.output_dir,
        num_steps=args.num_steps,
        train_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        mixed_precision=args.mixed_precision,
        save_every_n_steps=args.save_every,
        seed=args.seed,
        image_size=args.image_size,
    )
    trainer = InstructPix2PixTrainer(
        unet=unet,
        vae=vae,
        text_encoder=text_encoder,
        noise_scheduler=noise_scheduler,
        tokenizer=tokenizer,
        train_dataloader=train_dl,
        val_dataloader=val_dl,
        config=trainer_config,
        device=device,
    )
    stats = trainer.train()
    logger.info("Training complete: %s", stats)


if __name__ == "__main__":
    main()
