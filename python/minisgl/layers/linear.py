from __future__ import annotations

from typing import List

import torch
import torch.nn.functional as F
from minisgl.distributed import DistributedCommunicator, get_tp_info
from minisgl.quantization import (
    apply_w8a8_int8_linear,
    is_w8a8_int8_full_linear_enabled,
    quantize_weight_per_channel_int8,
)
from minisgl.utils import div_even

from .base import BaseOP


class _LinearTPImpl(BaseOP):
    """Real implementation of a linear layer with tensor parallelism."""

    def __init__(
        self,
        full_isize: int,
        full_osize: int,
        local_isize: int,
        local_osize: int,
        has_bias: bool,
    ):
        self.full_input_size = full_isize
        self.full_output_size = full_osize
        self.local_input_size = local_isize
        self.local_output_size = local_osize
        self.weight = torch.empty(local_osize, local_isize)
        self.bias = torch.empty(local_osize) if has_bias else None
        self.weight_scale: torch.Tensor | None = None
        # For stacked params loading (qkv_proj, gate_up_proj)
        self._stacked_params = {}

    def weight_loader(self, loaded_weight: torch.Tensor, shard_id=None):
        """Load weight directly into the parameter.

        Args:
            loaded_weight: The weight tensor to load
            shard_id: For stacked params, this identifies which shard (e.g., "q", "k", "v" for qkv,
                     or 0, 1 for gate_up_proj)
        """
        if shard_id is None:
            # Direct load
            # If self.weight is a meta tensor, replace it directly instead of copy
            if self.weight.is_meta:
                self.weight = loaded_weight
            else:
                self.weight.copy_(loaded_weight)
        else:
            # Stacked params: store the shard for later merging
            self._stacked_params[shard_id] = loaded_weight
            # Check if all shards are loaded
            self._maybe_merge_stacked_params()

    def _maybe_merge_stacked_params(self):
        """Merge stacked params when all shards are loaded."""
        # Subclasses can override this to handle specific merging logic
        pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.weight.dtype == torch.int8:
            assert self.weight_scale is not None
            return apply_w8a8_int8_linear(x, self.weight, self.weight_scale, self.bias)
        return F.linear(x, self.weight, self.bias)

    def process_weights_after_loading(self) -> None:
        if not is_w8a8_int8_full_linear_enabled() or self.weight.dtype == torch.int8:
            return
        self.weight, self.weight_scale = quantize_weight_per_channel_int8(self.weight)


class LinearReplicated(_LinearTPImpl):
    """
    Linear layer where weights are replicated (not sharded) across all TP ranks.
    Each GPU holds the full weight matrix.
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        has_bias: bool,
    ):
        super().__init__(
            full_isize=input_size,
            full_osize=output_size,
            local_isize=input_size,
            local_osize=output_size,
            has_bias=has_bias,
        )


class LinearColParallelMerged(_LinearTPImpl):
    def __init__(
        self,
        input_size: int,
        output_sizes: List[int],
        has_bias: bool,
    ):
        # check that all output sizes are divisible by tp_size
        tp_info = get_tp_info()
        tp_output_sizes = [div_even(size, tp_info.size) for size in output_sizes]
        output_size = sum(output_sizes)
        tp_output_size = sum(tp_output_sizes)
        self.output_sizes = output_sizes  # Store for merging
        self.tp_output_sizes = tp_output_sizes
        super().__init__(input_size, output_size, input_size, tp_output_size, has_bias)

    def weight_loader(self, loaded_weight: torch.Tensor, shard_id=None):
        """Load weight for merged linear layer (e.g., gate_up_proj, qkv_proj).

        Args:
            loaded_weight: The weight tensor to load
            shard_id: For stacked params, this identifies which shard:
                     - For gate_up_proj: 0 (gate) or 1 (up)
                     - For qkv_proj: "q", "k", or "v"
        """
        tp_info = get_tp_info()
        r = tp_info.rank
        n = tp_info.size

        if shard_id is None:
            # Direct load - the weight should already be the correct size
            # Handle TP sharding: take the shard for this rank
            if loaded_weight.shape[0] == self.full_output_size:
                # Not sharded, need to shard
                loaded_weight = loaded_weight.chunk(n, dim=0)[r]
            # If self.weight is a meta tensor, replace it directly instead of copy
            if self.weight.is_meta:
                self.weight = loaded_weight
            else:
                self.weight.copy_(loaded_weight)
        else:
            # Stacked params: store the shard for later merging
            self._stacked_params[shard_id] = loaded_weight
            self._maybe_merge_stacked_params()

    def _maybe_merge_stacked_params(self):
        """Merge stacked params when all shards are loaded."""
        # Check if we have all shards
        expected_shards = len(self.output_sizes)
        if len(self._stacked_params) < expected_shards:
            return

        # We have all shards, merge them
        tp_info = get_tp_info()
        r = tp_info.rank
        n = tp_info.size

        shards = []
        for i in range(expected_shards):
            shard = self._stacked_params[i]
            # Apply TP sharding
            if shard.shape[0] == self.full_output_size // expected_shards * n:
                # Already sharded, take this rank's portion
                shard = shard.chunk(n, dim=0)[r]
            shards.append(shard)

        # Concatenate along output dimension
        merged = torch.cat(shards, dim=0)
        # If self.weight is a meta tensor, replace it directly instead of copy
        if self.weight.is_meta:
            self.weight = merged
        else:
            self.weight.copy_(merged)
        self._stacked_params = {}


class LinearQKVMerged(_LinearTPImpl):
    def __init__(
        self,
        hidden_size: int,
        head_dim: int,
        num_qo_heads: int,
        num_kv_heads: int,
        has_bias: bool,
    ):
        tp_info = get_tp_info()

        GQA_ratio = div_even(num_qo_heads, num_kv_heads)
        local_num_kv = div_even(num_kv_heads, tp_info.size)
        full_isize = hidden_size
        full_osize = (GQA_ratio + 2) * num_kv_heads * head_dim
        local_isize = hidden_size
        local_osize = (GQA_ratio + 2) * local_num_kv * head_dim
        self.shard_ids = ["q", "k", "v"]  # Store for merging
        super().__init__(full_isize, full_osize, local_isize, local_osize, has_bias)

    def weight_loader(self, loaded_weight: torch.Tensor, shard_id=None):
        """Load weight for QKV merged linear layer.

        Args:
            loaded_weight: The weight tensor to load
            shard_id: "q", "k", or "v"
        """
        tp_info = get_tp_info()
        r = tp_info.rank
        n = tp_info.size

        if shard_id is None:
            # Direct load
            if loaded_weight.shape[0] == self.full_output_size:
                loaded_weight = loaded_weight.chunk(n, dim=0)[r]
            # If self.weight is a meta tensor, replace it directly instead of copy
            if self.weight.is_meta:
                self.weight = loaded_weight
            else:
                self.weight.copy_(loaded_weight)
        else:
            # Store for merging
            self._stacked_params[shard_id] = loaded_weight
            self._maybe_merge_stacked_params()

    def _maybe_merge_stacked_params(self):
        """Merge QKV when all shards are loaded."""
        if len(self._stacked_params) < len(self.shard_ids):
            return

        tp_info = get_tp_info()
        r = tp_info.rank
        n = tp_info.size

        shards = []
        for sid in self.shard_ids:
            shard = self._stacked_params[sid]
            # Each shard has full output for its portion
            # Shard by tp rank
            shard = shard.chunk(n, dim=0)[r]
            shards.append(shard)

        merged = torch.cat(shards, dim=0)
        # If self.weight is a meta tensor, replace it directly instead of copy
        if self.weight.is_meta:
            self.weight = merged
        else:
            self.weight.copy_(merged)
        self._stacked_params = {}


class LinearQKVMerged(_LinearTPImpl):
    def __init__(
        self,
        hidden_size: int,
        head_dim: int,
        num_qo_heads: int,
        num_kv_heads: int,
        has_bias: bool,
    ):
        tp_info = get_tp_info()

        GQA_ratio = div_even(num_qo_heads, num_kv_heads)
        local_num_kv = div_even(num_kv_heads, tp_info.size)
        full_isize = hidden_size
        full_osize = (GQA_ratio + 2) * num_kv_heads * head_dim
        local_isize = hidden_size
        local_osize = (GQA_ratio + 2) * local_num_kv * head_dim
        super().__init__(full_isize, full_osize, local_isize, local_osize, has_bias)


class LinearOProj(_LinearTPImpl):
    def __init__(self, input_size: int, output_size: int, has_bias: bool):
        tp_info = get_tp_info()
        full_isize = input_size
        full_osize = output_size
        local_isize = div_even(input_size, tp_info.size)
        local_osize = output_size
        self._comm = DistributedCommunicator()
        self._tp_size = tp_info.size
        super().__init__(full_isize, full_osize, local_isize, local_osize, has_bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = super().forward(x)
        if self._tp_size > 1:
            y = self._comm.all_reduce(y)
        return y

    def weight_loader(self, loaded_weight: torch.Tensor, shard_id=None):
        if shard_id is not None:
            return super().weight_loader(loaded_weight, shard_id=shard_id)
        if loaded_weight.shape[1] == self.full_input_size:
            loaded_weight = loaded_weight.chunk(self._tp_size, dim=1)[get_tp_info().rank].contiguous()
        if self.weight.is_meta:
            self.weight = loaded_weight
        else:
            self.weight.copy_(loaded_weight)


class LinearRowParallel(_LinearTPImpl):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        has_bias: bool,
    ):
        tp_info = get_tp_info()
        local_input_size = div_even(input_size, tp_info.size)
        local_output_size = output_size
        self._comm = DistributedCommunicator()
        self._tp_size = tp_info.size
        super().__init__(input_size, output_size, local_input_size, local_output_size, has_bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = super().forward(x)
        if self._tp_size > 1:
            y = self._comm.all_reduce(y)
        return y

    def weight_loader(self, loaded_weight: torch.Tensor, shard_id=None):
        if shard_id is not None:
            return super().weight_loader(loaded_weight, shard_id=shard_id)
        if loaded_weight.shape[1] == self.full_input_size:
            loaded_weight = loaded_weight.chunk(self._tp_size, dim=1)[get_tp_info().rank].contiguous()
        if self.weight.is_meta:
            self.weight = loaded_weight
        else:
            self.weight.copy_(loaded_weight)
