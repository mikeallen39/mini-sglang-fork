from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from minisgl.layers import BaseOP

if TYPE_CHECKING:
    import torch


class BaseLLMModel(ABC, BaseOP):
    @abstractmethod
    def forward(self) -> torch.Tensor: ...

    @property
    def supports_prefix_cache(self) -> bool:
        return True

    @property
    def supports_cuda_graph(self) -> bool:
        return True

    def clear_runtime_state_slot(self, table_idx: int) -> None:
        _ = table_idx
