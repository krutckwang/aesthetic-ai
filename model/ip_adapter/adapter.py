"""IP-Adapter wrapper for identity preservation at inference.

IP-Adapter conditions image generation on a reference face embedding extracted by
a CLIP image encoder. At inference, the reference face (e.g., the before-image
cropped to the face region) is encoded and injected via cross-attention, ensuring
the generated face retains identity even after the treatment is applied.

Scale controls the strength of the identity conditioning (0.0 = off, 1.0 = full).
Recommended: 0.8 for aesthetic treatments (preserves identity while allowing change).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn


@dataclass
class IPAdapterConfig:
    scale: float = 0.8           # cross-attention weight for identity conditioning
    num_tokens: int = 4          # number of image prompt tokens (IP-Adapter default: 4)
    image_encoder_id: str = "openai/clip-vit-large-patch14"

    @classmethod
    def from_dict(cls, d: dict) -> "IPAdapterConfig":
        return cls(
            scale=d.get("scale", 0.8),
            num_tokens=d.get("num_tokens", 4),
            image_encoder_id=d.get("image_encoder_id", cls.image_encoder_id),
        )


class ImageProjection(nn.Module):
    """Projects CLIP image embeddings to the UNet cross-attention key/value space."""

    def __init__(self, clip_dim: int = 768, cross_attention_dim: int = 768, num_tokens: int = 4) -> None:
        super().__init__()
        self.num_tokens = num_tokens
        self.image_embeds = nn.Linear(clip_dim, cross_attention_dim * num_tokens)
        self.norm = nn.LayerNorm(cross_attention_dim)

    def forward(self, image_embeds: torch.Tensor) -> torch.Tensor:
        # image_embeds: (B, clip_dim) → (B, num_tokens, cross_attention_dim)
        projected = self.image_embeds(image_embeds)
        projected = projected.reshape(-1, self.num_tokens, projected.shape[-1] // self.num_tokens)
        return self.norm(projected)


class IPAdapter:
    """
    Wraps a diffusion pipeline to add IP-Adapter identity conditioning.

    Usage (inference):
        adapter = IPAdapter(pipeline, config)
        adapter.set_scale(0.8)
        images = adapter.generate(
            reference_image=before_face_crop,
            prompt="Apply botulinum toxin ...",
            num_inference_steps=30,
        )
    """

    def __init__(self, pipeline: Any, config: IPAdapterConfig | None = None) -> None:
        self.pipeline = pipeline
        self.config = config or IPAdapterConfig()
        self._projection: ImageProjection | None = None
        self._loaded = False

    # ── Public ────────────────────────────────────────────────────────────────

    def set_scale(self, scale: float) -> None:
        self.config = IPAdapterConfig(
            scale=scale,
            num_tokens=self.config.num_tokens,
            image_encoder_id=self.config.image_encoder_id,
        )

    def load_weights(self, weights_path: str | Path) -> None:
        """Load pretrained IP-Adapter projection weights."""
        state = torch.load(str(weights_path), map_location="cpu")
        if self._projection is None:
            self._projection = ImageProjection()
        self._projection.load_state_dict(state, strict=False)
        self._projection.eval()
        self._loaded = True

    def encode_reference_image(self, image) -> torch.Tensor:
        """Extract CLIP embedding from a reference image (PIL or tensor)."""
        from PIL import Image as PILImage
        processor = self._get_image_processor()
        inputs = processor(images=image, return_tensors="pt")
        with torch.no_grad():
            encoder = self._get_image_encoder()
            embeds = encoder(**inputs).image_embeds
        return embeds

    def get_image_tokens(self, reference_image) -> torch.Tensor:
        """Return projected image tokens ready for UNet cross-attention injection."""
        if self._projection is None:
            raise RuntimeError("Call load_weights() before get_image_tokens()")
        embeds = self.encode_reference_image(reference_image)
        return self._projection(embeds)

    # ── Private ───────────────────────────────────────────────────────────────

    def _get_image_encoder(self):
        from transformers import CLIPVisionModelWithProjection
        if not hasattr(self, "_image_encoder"):
            self._image_encoder = CLIPVisionModelWithProjection.from_pretrained(
                self.config.image_encoder_id
            )
        return self._image_encoder

    def _get_image_processor(self):
        from transformers import CLIPImageProcessor
        if not hasattr(self, "_image_processor"):
            self._image_processor = CLIPImageProcessor.from_pretrained(
                self.config.image_encoder_id
            )
        return self._image_processor
