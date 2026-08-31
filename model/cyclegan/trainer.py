"""CycleGAN training loop (baseline comparison)."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from model.cyclegan.model import CycleGAN

logger = logging.getLogger(__name__)


@dataclass
class CycleGANTrainerConfig:
    output_dir: str = "outputs/cyclegan"
    num_steps: int = 50_000
    train_batch_size: int = 1
    learning_rate_g: float = 2e-4
    learning_rate_d: float = 2e-4
    beta1: float = 0.5
    beta2: float = 0.999
    lambda_cycle: float = 10.0
    lambda_identity: float = 0.5
    save_every_n_steps: int = 1000
    log_every_n_steps: int = 100

    @classmethod
    def from_dict(cls, d: dict) -> "CycleGANTrainerConfig":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


class CycleGANTrainer:
    """Trains a CycleGAN model for before→after domain translation."""

    def __init__(
        self,
        model: CycleGAN,
        train_dataloader: DataLoader,
        config: CycleGANTrainerConfig,
        device: str | torch.device = "cuda",
    ) -> None:
        self.model = model.to(torch.device(device))
        self.dataloader = train_dataloader
        self.config = config
        self.device = torch.device(device)
        self._step = 0
        self._opt_G: torch.optim.Optimizer | None = None
        self._opt_D: torch.optim.Optimizer | None = None

    def setup(self) -> None:
        self._opt_G = torch.optim.Adam(
            self.model.generator_params(),
            lr=self.config.learning_rate_g,
            betas=(self.config.beta1, self.config.beta2),
        )
        self._opt_D = torch.optim.Adam(
            self.model.discriminator_params(),
            lr=self.config.learning_rate_d,
            betas=(self.config.beta1, self.config.beta2),
        )

    def train(self) -> dict[str, float]:
        self.setup()
        data_iter = _infinite(self.dataloader)
        total_g_loss = 0.0
        total_d_loss = 0.0

        while self._step < self.config.num_steps:
            batch = next(data_iter)
            real_A = batch["pixel_values"].to(self.device)
            real_B = batch["edited_pixel_values"].to(self.device)

            # Generator step
            self._opt_G.zero_grad()
            g_losses = self.model.compute_generator_loss(real_A, real_B)
            g_losses["total"].backward()
            self._opt_G.step()
            total_g_loss += g_losses["total"].item()

            # Discriminator step
            self._opt_D.zero_grad()
            d_losses = self.model.compute_discriminator_loss(real_A, real_B)
            d_losses["total"].backward()
            self._opt_D.step()
            total_d_loss += d_losses["total"].item()

            if self._step % self.config.log_every_n_steps == 0:
                logger.info(
                    "step=%d g_loss=%.4f d_loss=%.4f",
                    self._step, g_losses["total"].item(), d_losses["total"].item(),
                )

            if self._step % self.config.save_every_n_steps == 0 and self._step > 0:
                self._save_checkpoint()

            self._step += 1

        self._save_checkpoint(final=True)
        n = max(self._step, 1)
        return {"avg_g_loss": total_g_loss / n, "avg_d_loss": total_d_loss / n, "steps": self._step}

    def _save_checkpoint(self, final: bool = False) -> None:
        tag = "final" if final else f"step_{self._step}"
        out = Path(self.config.output_dir) / tag
        out.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), out / "cyclegan.pt")
        logger.info("Saved CycleGAN checkpoint to %s", out)


def _infinite(dataloader):
    while True:
        yield from dataloader
