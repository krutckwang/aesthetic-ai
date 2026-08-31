"""CycleGAN model components — used as baseline comparison against InstructPix2Pix.

Architecture:
  - Generator: ResNet-based (9 residual blocks for 256+ images)
  - Discriminator: PatchGAN (70×70 receptive field)
  - Two generator/discriminator pairs: G_AB (before→after), G_BA (after→before)
  - Loss: cycle-consistency + adversarial + identity
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Building blocks ───────────────────────────────────────────────────────────

class ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, kernel_size=3),
            nn.InstanceNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, kernel_size=3),
            nn.InstanceNorm2d(channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class ResNetGenerator(nn.Module):
    """ResNet generator for CycleGAN. Transforms domain A images to domain B."""

    def __init__(self, in_channels: int = 3, out_channels: int = 3,
                 base_filters: int = 64, n_residual_blocks: int = 9) -> None:
        super().__init__()
        layers: list[nn.Module] = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels, base_filters, kernel_size=7),
            nn.InstanceNorm2d(base_filters),
            nn.ReLU(inplace=True),
        ]
        # Downsampling
        filters = base_filters
        for _ in range(2):
            layers += [
                nn.Conv2d(filters, filters * 2, kernel_size=3, stride=2, padding=1),
                nn.InstanceNorm2d(filters * 2),
                nn.ReLU(inplace=True),
            ]
            filters *= 2
        # Residual blocks
        for _ in range(n_residual_blocks):
            layers.append(ResidualBlock(filters))
        # Upsampling
        for _ in range(2):
            layers += [
                nn.ConvTranspose2d(filters, filters // 2, kernel_size=3, stride=2,
                                   padding=1, output_padding=1),
                nn.InstanceNorm2d(filters // 2),
                nn.ReLU(inplace=True),
            ]
            filters //= 2
        layers += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(filters, out_channels, kernel_size=7),
            nn.Tanh(),
        ]
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class PatchGANDiscriminator(nn.Module):
    """70×70 PatchGAN discriminator. Returns a grid of real/fake logits."""

    def __init__(self, in_channels: int = 3, base_filters: int = 64, n_layers: int = 3) -> None:
        super().__init__()
        layers: list[nn.Module] = [
            nn.Conv2d(in_channels, base_filters, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        ]
        filters = base_filters
        for i in range(1, n_layers):
            stride = 2 if i < n_layers - 1 else 1
            layers += [
                nn.Conv2d(filters, filters * 2, kernel_size=4, stride=stride, padding=1),
                nn.InstanceNorm2d(filters * 2),
                nn.LeakyReLU(0.2, inplace=True),
            ]
            filters *= 2
        layers.append(nn.Conv2d(filters, 1, kernel_size=4, stride=1, padding=1))
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


# ── Full CycleGAN ─────────────────────────────────────────────────────────────

class CycleGAN(nn.Module):
    """
    Full CycleGAN model with two generator/discriminator pairs.

    Domain A = before-treatment images
    Domain B = after-treatment images
    """

    def __init__(
        self,
        base_filters: int = 64,
        n_residual_blocks: int = 9,
        lambda_cycle: float = 10.0,
        lambda_identity: float = 0.5,
    ) -> None:
        super().__init__()
        self.lambda_cycle = lambda_cycle
        self.lambda_identity = lambda_identity

        self.G_AB = ResNetGenerator(3, 3, base_filters, n_residual_blocks)
        self.G_BA = ResNetGenerator(3, 3, base_filters, n_residual_blocks)
        self.D_A = PatchGANDiscriminator(3, base_filters)
        self.D_B = PatchGANDiscriminator(3, base_filters)

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.normal_(m.weight, 0.0, 0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def generator_params(self):
        return list(self.G_AB.parameters()) + list(self.G_BA.parameters())

    def discriminator_params(self):
        return list(self.D_A.parameters()) + list(self.D_B.parameters())

    def compute_generator_loss(
        self, real_A: torch.Tensor, real_B: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        fake_B = self.G_AB(real_A)
        fake_A = self.G_BA(real_B)
        rec_A = self.G_BA(fake_B)
        rec_B = self.G_AB(fake_A)

        # Adversarial losses
        loss_adv_AB = F.mse_loss(self.D_B(fake_B), torch.ones_like(self.D_B(fake_B)))
        loss_adv_BA = F.mse_loss(self.D_A(fake_A), torch.ones_like(self.D_A(fake_A)))

        # Cycle-consistency losses
        loss_cycle_A = F.l1_loss(rec_A, real_A) * self.lambda_cycle
        loss_cycle_B = F.l1_loss(rec_B, real_B) * self.lambda_cycle

        # Identity losses
        idt_A = self.G_BA(real_A)
        idt_B = self.G_AB(real_B)
        loss_idt_A = F.l1_loss(idt_A, real_A) * self.lambda_identity
        loss_idt_B = F.l1_loss(idt_B, real_B) * self.lambda_identity

        total = loss_adv_AB + loss_adv_BA + loss_cycle_A + loss_cycle_B + loss_idt_A + loss_idt_B
        return {
            "total": total,
            "adv_AB": loss_adv_AB,
            "adv_BA": loss_adv_BA,
            "cycle_A": loss_cycle_A,
            "cycle_B": loss_cycle_B,
            "idt_A": loss_idt_A,
            "idt_B": loss_idt_B,
        }

    def compute_discriminator_loss(
        self, real_A: torch.Tensor, real_B: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        with torch.no_grad():
            fake_B = self.G_AB(real_A)
            fake_A = self.G_BA(real_B)

        real_label = torch.ones
        fake_label = torch.zeros

        loss_D_A = 0.5 * (
            F.mse_loss(self.D_A(real_A), torch.ones_like(self.D_A(real_A))) +
            F.mse_loss(self.D_A(fake_A), torch.zeros_like(self.D_A(fake_A)))
        )
        loss_D_B = 0.5 * (
            F.mse_loss(self.D_B(real_B), torch.ones_like(self.D_B(real_B))) +
            F.mse_loss(self.D_B(fake_B), torch.zeros_like(self.D_B(fake_B)))
        )
        return {"total": loss_D_A + loss_D_B, "D_A": loss_D_A, "D_B": loss_D_B}
