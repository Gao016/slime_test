"""Global registry for teacher lm_head weights.

Cached once at checkpoint load time as CPU pinned tensors.
Accessed by loss computation which does on-demand H2D transfer.
Supports multi-teacher via distinct tag keys.
"""

from __future__ import annotations

import torch


class TeacherHeadRegistry:
    _weights: dict[str, torch.Tensor] = {}

    @classmethod
    def set(cls, tag: str, weight: torch.Tensor) -> None:
        cls._weights[tag] = weight.to("cpu").pin_memory()

    @classmethod
    def get(cls, tag: str) -> torch.Tensor | None:
        return cls._weights.get(tag)

    @classmethod
    def clear(cls) -> None:
        cls._weights.clear()
