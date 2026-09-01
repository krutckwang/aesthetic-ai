"""InstructPix2Pix + LoRA training loop.

Trains the UNet with LoRA adapters on the aesthetic before/after dataset.
IP-Adapter weights are NOT trained here — they are loaded at inference only.

Key hyperparameters (from training plan):
  - 15,000 steps
  - fp16 mixed precision
  - AdamW, lr=1e-4, weight_decay=0.01
  - Gradient clipping: max_norm=1.0
  - Noise offset: 0.05 (prevents colour shift on dark images)
  - Conditioning dropout: 10% (enables classifier-free guidance)
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


@dataclass
class TrainerConfig:
    output_dir: str = "outputs/lora"
    num_steps: int = 15_000
    train_batch_size: int = 4
    gradient_accumulation_steps: int = 1
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    mixed_precision: str = "fp16"        # "fp16", "bf16", or "no"
    noise_offset: float = 0.05
    conditioning_dropout_prob: float = 0.10
    save_every_n_steps: int = 500
    log_every_n_steps: int = 50
    seed: int = 42
    image_size: int = 512

    @classmethod
    def from_dict(cls, d: dict) -> "TrainerConfig":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


class InstructPix2PixTrainer:
    """Manages the fine-tuning loop for InstructPix2Pix + LoRA."""

    def __init__(
        self,
        unet,
        vae,
        text_encoder,
        noise_scheduler,
        tokenizer,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader | None,
        config: TrainerConfig,
        device: str | torch.device = "cuda",
    ) -> None:
        self.unet = unet
        self.vae = vae
        self.text_encoder = text_encoder
        self.noise_scheduler = noise_scheduler
        self.tokenizer = tokenizer
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.config = config
        self.device = torch.device(device)
        self._step = 0
        self._optimizer: torch.optim.Optimizer | None = None
        self._scaler: torch.cuda.amp.GradScaler | None = None

    # ── Public ────────────────────────────────────────────────────────────────

    def setup(self) -> None:
        """Freeze non-LoRA weights, build optimizer and scaler."""
        trainable = [p for p in self.unet.parameters() if p.requires_grad]
        self._optimizer = torch.optim.AdamW(
            trainable,
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        if self.config.mixed_precision == "fp16":
            self._scaler = torch.amp.GradScaler('cuda')

        # Freeze VAE and text encoder — never trained
        for model in (self.vae, self.text_encoder):
            for p in model.parameters():
                p.requires_grad = False

    def train(self) -> dict[str, float]:
        """Run the full training loop. Returns final loss stats."""
        self.setup()
        self.unet.train()
        self.vae.eval()
        self.text_encoder.eval()

        total_loss = 0.0
        data_iter = _infinite(self.train_dataloader)

        while self._step < self.config.num_steps:
            batch = next(data_iter)
            loss = self._training_step(batch)
            total_loss += loss

            if self._step % self.config.log_every_n_steps == 0:
                logger.info("step=%d loss=%.4f", self._step, loss)

            if self._step % self.config.save_every_n_steps == 0 and self._step > 0:
                self._save_checkpoint()

            self._step += 1

        self._save_checkpoint(final=True)
        avg_loss = total_loss / max(self._step, 1)
        return {"avg_loss": avg_loss, "steps": self._step}

    # ── Private ───────────────────────────────────────────────────────────────

    def _training_step(self, batch: dict) -> float:
        pixel_values = batch["pixel_values"].to(self.device)
        edited_pixel_values = batch["edited_pixel_values"].to(self.device)
        input_ids = batch.get("input_ids")

        # Encode images to latent space
        with torch.no_grad():
            before_latents = self._encode(pixel_values)
            after_latents = self._encode(edited_pixel_values)

        # Sample noise with offset
        noise = torch.randn_like(after_latents)
        if self.config.noise_offset > 0:
            noise += self.config.noise_offset * torch.randn(
                after_latents.shape[0], after_latents.shape[1], 1, 1,
                device=self.device,
            )

        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps,
            (after_latents.shape[0],), device=self.device,
        ).long()

        noisy_latents = self.noise_scheduler.add_noise(after_latents, noise, timesteps)

        # InstructPix2Pix conditioning: concat before_latents + noisy_latents on channel dim
        latent_model_input = torch.cat([noisy_latents, before_latents], dim=1)

        # Conditioning dropout — drop text or image conditioning randomly
        encoder_hidden_states = self._get_encoder_hidden_states(input_ids, pixel_values)

        use_fp16 = self.config.mixed_precision == "fp16"
        with torch.amp.autocast("cuda", enabled=use_fp16):
            noise_pred = self.unet(latent_model_input, timesteps, encoder_hidden_states).sample
            loss = F.mse_loss(noise_pred.float(), noise.float(), reduction="mean")

        self._backward(loss)
        return loss.item()

    def _encode(self, images: torch.Tensor) -> torch.Tensor:
        latents = self.vae.encode(images.to(dtype=self.vae.dtype)).latent_dist.sample()
        return latents * self.vae.config.scaling_factor

    def _get_encoder_hidden_states(self, input_ids, pixel_values) -> torch.Tensor:
        with torch.no_grad():
            if input_ids is not None:
                ids = input_ids.to(self.device)
                # Conditioning dropout: randomly zero out text tokens
                if self.config.conditioning_dropout_prob > 0:
                    mask = (torch.rand(ids.shape[0]) > self.config.conditioning_dropout_prob)
                    mask = mask.to(self.device)
                    ids = ids * mask.unsqueeze(1)
                return self.text_encoder(ids)[0]
            else:
                # Fallback: empty conditioning
                bs = pixel_values.shape[0]
                return torch.zeros(bs, 1, self.text_encoder.config.hidden_size, device=self.device)

    def _backward(self, loss: torch.Tensor) -> None:
        if self._scaler is not None:
            self._scaler.scale(loss).backward()
            self._scaler.unscale_(self._optimizer)
            torch.nn.utils.clip_grad_norm_(
                [p for p in self.unet.parameters() if p.requires_grad],
                self.config.max_grad_norm,
            )
            self._scaler.step(self._optimizer)
            self._scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in self.unet.parameters() if p.requires_grad],
                self.config.max_grad_norm,
            )
            self._optimizer.step()
        self._optimizer.zero_grad()

    def _save_checkpoint(self, final: bool = False) -> None:
        tag = "final" if final else f"step_{self._step}"
        out = Path(self.config.output_dir) / tag
        out.mkdir(parents=True, exist_ok=True)

        lora_state = {
            k: v for k, v in self.unet.state_dict().items()
            if "lora_A" in k or "lora_B" in k
        }
        torch.save(lora_state, out / "lora_weights.pt")
        logger.info("Saved checkpoint to %s", out)


def _infinite(dataloader: DataLoader):
    """Cycle a dataloader indefinitely."""
    while True:
        yield from dataloader
