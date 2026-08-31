"""Evaluator — runs metrics over a full val/test split and produces a report."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from model.evaluation.metrics import MetricTracker, evaluate_batch, MetricResult

logger = logging.getLogger(__name__)


@dataclass
class EvaluationReport:
    model_tag: str
    num_samples: int
    metrics: dict[str, float]
    per_treatment: dict[str, dict[str, float]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_tag": self.model_tag,
            "num_samples": self.num_samples,
            "metrics": self.metrics,
            "per_treatment": self.per_treatment,
        }

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        logger.info("Evaluation report saved to %s", path)


class Evaluator:
    """
    Runs batch evaluation of a generative model against a DataLoader.

    The model is expected to implement:
        model.generate(pixel_values) -> torch.Tensor  (same shape, [-1, 1])
    """

    def __init__(self, model: Any, model_tag: str, device: str | torch.device = "cpu") -> None:
        self.model = model
        self.model_tag = model_tag
        self.device = torch.device(device)

    def evaluate(self, dataloader: DataLoader) -> EvaluationReport:
        tracker = MetricTracker()
        per_treatment: dict[str, MetricTracker] = {}
        n = 0

        self.model.eval()
        with torch.no_grad():
            for batch in dataloader:
                pred = self._generate(batch)
                target = batch["edited_pixel_values"].to(self.device)
                result = evaluate_batch(pred, target)
                tracker.update(result)
                n += pred.shape[0]

                # Per-treatment breakdown
                if "treatment_category" in batch:
                    for i, treatment in enumerate(batch["treatment_category"]):
                        t = treatment or "unknown"
                        if t not in per_treatment:
                            per_treatment[t] = MetricTracker()
                        single = evaluate_batch(
                            pred[i: i + 1], target[i: i + 1]
                        )
                        per_treatment[t].update(single)

                logger.debug("Evaluated %d samples", n)

        per_treatment_avg = {t: tr.compute() for t, tr in per_treatment.items()}
        return EvaluationReport(
            model_tag=self.model_tag,
            num_samples=n,
            metrics=tracker.compute(),
            per_treatment=per_treatment_avg,
        )

    def _generate(self, batch: dict) -> torch.Tensor:
        pixel_values = batch["pixel_values"].to(self.device)
        instruction = batch.get("instruction", [""] * pixel_values.shape[0])
        if hasattr(self.model, "generate"):
            return self.model.generate(pixel_values, instruction)
        # Fallback: identity (returns before image as prediction — worst case baseline)
        return pixel_values
