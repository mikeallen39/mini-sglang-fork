from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from minisgl.layers import BaseOP

if TYPE_CHECKING:
    import torch
    from minisgl.core import Batch, Req


class BaseLLMModel(ABC, BaseOP):
    @abstractmethod
    def forward(self) -> torch.Tensor: ...

    @property
    def supports_prefix_cache(self) -> bool:
        return True

    @property
    def supports_cuda_graph(self) -> bool:
        return True

    def prepare_for_cuda_graph_replay(self, batch: Batch, dummy_reqs: list[Req]) -> None:
        _ = batch, dummy_reqs

    def finish_cuda_graph_replay(self, batch: Batch, dummy_reqs: list[Req]) -> None:
        _ = batch, dummy_reqs

    def clear_runtime_state_slot(self, table_idx: int) -> None:
        _ = table_idx

    def copy_runtime_state_slot(self, src_table_idx: int, dst_table_idx: int) -> None:
        _ = src_table_idx, dst_table_idx
