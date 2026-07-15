from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Iterable, Mapping


GROUP_NAMES = ("clean", "valid", "single", "invalid")
GROUP_TO_ID = {name: idx for idx, name in enumerate(GROUP_NAMES)}
ID_TO_GROUP = {idx: name for name, idx in GROUP_TO_ID.items()}


@dataclass
class FeedbackAdaptiveController:
    """Feedback controller for four CCDT loss components.

    The controller never reads test-set metrics. It is updated from a fixed
    feedback subset sampled from the training split. The four weights are
    normalized to keep their sum equal to four, so an all-ones vector exactly
    recovers the original fixed-weight objective.
    """

    mode: str = "feedback"
    ema: float = 0.9
    temperature: float = 1.0
    min_weight: float = 0.25
    max_weight: float = 2.50
    warmup_epochs: int = 5
    target_valid_asr: float = 0.99
    target_single_leak: float = 0.02
    target_invalid_leak: float = 0.02
    target_wrong_asr: float = 0.01
    clean_tolerance: float = 0.01
    asr_tolerance: float = 0.01
    leak_tolerance: float = 0.02
    wrong_tolerance: float = 0.01
    max_score: float = 4.0
    manual_weights: Mapping[str, float] | None = None
    weights: Dict[str, float] = field(default_factory=lambda: {name: 1.0 for name in GROUP_NAMES})
    clean_reference: float | None = None

    def __post_init__(self) -> None:
        valid_modes = {"equal", "manual", "feedback"}
        if self.mode not in valid_modes:
            raise ValueError(f"Unknown adaptive mode={self.mode!r}; valid={sorted(valid_modes)}")
        if not (0.0 <= self.ema < 1.0):
            raise ValueError("ema must be in [0, 1)")
        if self.temperature <= 0:
            raise ValueError("temperature must be positive")
        if self.min_weight <= 0 or self.max_weight <= 0:
            raise ValueError("weight bounds must be positive")
        if self.min_weight > self.max_weight:
            raise ValueError("min_weight must not exceed max_weight")

        if self.mode == "manual":
            if self.manual_weights is None:
                self.manual_weights = {
                    "clean": 1.0,
                    "valid": 1.5,
                    "single": 0.75,
                    "invalid": 1.25,
                }
            self.weights = self._normalize_and_clip(dict(self.manual_weights))
        else:
            self.weights = {name: 1.0 for name in GROUP_NAMES}

    def state_dict(self) -> Dict[str, object]:
        return {
            "mode": self.mode,
            "ema": self.ema,
            "temperature": self.temperature,
            "min_weight": self.min_weight,
            "max_weight": self.max_weight,
            "warmup_epochs": self.warmup_epochs,
            "target_valid_asr": self.target_valid_asr,
            "target_single_leak": self.target_single_leak,
            "target_invalid_leak": self.target_invalid_leak,
            "target_wrong_asr": self.target_wrong_asr,
            "clean_tolerance": self.clean_tolerance,
            "asr_tolerance": self.asr_tolerance,
            "leak_tolerance": self.leak_tolerance,
            "wrong_tolerance": self.wrong_tolerance,
            "max_score": self.max_score,
            "weights": dict(self.weights),
            "clean_reference": self.clean_reference,
        }

    def current_weights(self) -> Dict[str, float]:
        return dict(self.weights)

    def update(self, epoch: int, feedback_metrics: Mapping[str, float]) -> Dict[str, object]:
        clean_acc = float(feedback_metrics["clean_acc"])
        valid_asr = float(feedback_metrics["valid_asr"])
        wrong_asr = float(feedback_metrics["wrong_asr"])
        single_leak = float(feedback_metrics["single_leak"])
        invalid_leak = float(feedback_metrics["invalid_leak"])

        if self.clean_reference is None:
            self.clean_reference = clean_acc

        old_weights = dict(self.weights)

        if self.mode == "equal":
            self.weights = {name: 1.0 for name in GROUP_NAMES}
            self.clean_reference = max(float(self.clean_reference), clean_acc)
            return self._report(epoch, feedback_metrics, old_weights, {}, "equal")

        if self.mode == "manual":
            self.clean_reference = max(float(self.clean_reference), clean_acc)
            return self._report(epoch, feedback_metrics, old_weights, {}, "manual")

        if epoch <= self.warmup_epochs:
            self.weights = {name: 1.0 for name in GROUP_NAMES}
            self.clean_reference = max(float(self.clean_reference), clean_acc)
            return self._report(epoch, feedback_metrics, old_weights, {}, "warmup")

        clean_score = max(
            0.0,
            (float(self.clean_reference) - clean_acc) / max(self.clean_tolerance, 1e-12),
        )
        valid_deficit = max(
            0.0,
            (self.target_valid_asr - valid_asr) / max(self.asr_tolerance, 1e-12),
        )
        wrong_violation = max(
            0.0,
            (wrong_asr - self.target_wrong_asr) / max(self.wrong_tolerance, 1e-12),
        )
        valid_score = max(valid_deficit, wrong_violation)
        single_score = max(
            0.0,
            (single_leak - self.target_single_leak) / max(self.leak_tolerance, 1e-12),
        )
        invalid_score = max(
            0.0,
            (invalid_leak - self.target_invalid_leak) / max(self.leak_tolerance, 1e-12),
        )

        scores = {
            "clean": min(clean_score, self.max_score),
            "valid": min(valid_score, self.max_score),
            "single": min(single_score, self.max_score),
            "invalid": min(invalid_score, self.max_score),
        }

        raw = self._softmax_weights(scores)
        smoothed = {
            name: self.ema * old_weights[name] + (1.0 - self.ema) * raw[name]
            for name in GROUP_NAMES
        }
        self.weights = self._normalize_and_clip(smoothed)
        self.clean_reference = max(float(self.clean_reference), clean_acc)

        return self._report(epoch, feedback_metrics, old_weights, scores, "updated")

    def _softmax_weights(self, scores: Mapping[str, float]) -> Dict[str, float]:
        scaled = {name: float(scores[name]) / self.temperature for name in GROUP_NAMES}
        max_scaled = max(scaled.values())
        exp_values = {name: math.exp(scaled[name] - max_scaled) for name in GROUP_NAMES}
        total = sum(exp_values.values())
        return {name: 4.0 * exp_values[name] / total for name in GROUP_NAMES}

    def _normalize_and_clip(self, values: Mapping[str, float]) -> Dict[str, float]:
        current = {name: float(values[name]) for name in GROUP_NAMES}
        for _ in range(4):
            total = sum(current.values())
            if total <= 0:
                current = {name: 1.0 for name in GROUP_NAMES}
                total = 4.0
            current = {name: 4.0 * value / total for name, value in current.items()}
            current = {
                name: min(self.max_weight, max(self.min_weight, value))
                for name, value in current.items()
            }
        total = sum(current.values())
        return {name: 4.0 * value / total for name, value in current.items()}

    def _report(
        self,
        epoch: int,
        feedback_metrics: Mapping[str, float],
        old_weights: Mapping[str, float],
        scores: Mapping[str, float],
        status: str,
    ) -> Dict[str, object]:
        return {
            "epoch": int(epoch),
            "status": status,
            "old_weights": dict(old_weights),
            "new_weights": dict(self.weights),
            "scores": dict(scores),
            "clean_reference": self.clean_reference,
            "feedback_metrics": {
                key: (float(value) if isinstance(value, (int, float)) and value is not None else value)
                for key, value in feedback_metrics.items()
            },
        }
