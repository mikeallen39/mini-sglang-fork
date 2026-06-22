"""
MLA (Multi-Latent Attention) Backend using FlashInfer's MLA API.

FlashInfer provides BatchMLAPagedAttentionWrapper which supports:
- q_nope: [num_tokens, num_heads, kv_lora_rank]
- q_pe: [num_tokens, num_heads, qk_rope_head_dim]
- ckv_cache: [cache_size, 1, kv_lora_rank]
- kpe_cache: [cache_size, 1, qk_rope_head_dim]

Reference:
- sglang/python/sglang/srt/layers/attention/flashinfer_mla_backend.py
- flashinfer.mla.BatchMLAPagedAttentionWrapper
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import cached_property
from typing import TYPE_CHECKING, Dict, List, Literal, Optional

import torch
from minisgl.core import Batch, get_global_ctx
from minisgl.distributed import get_tp_info
from minisgl.env import ENV
from minisgl.utils import div_even, init_logger

from .base import BaseAttnBackend, BaseAttnMetadata
from .utils import BaseCaptureData

if TYPE_CHECKING:
    from flashinfer.mla import BatchMLAPagedAttentionWrapper
    from minisgl.kvcache import MLAKVCache
    from minisgl.models import ModelConfig


def _next_power_of_2(n: int) -> int:
    if n <= 1:
        return 1
    return 1 << math.ceil(math.log2(n))


logger = init_logger(__name__)


@dataclass
class MLACaptureData(BaseCaptureData):
    @property
    def one_tensor(self) -> torch.Tensor:
        return self.seq_lens

    @property
    def indices(self) -> torch.Tensor:
        return self.page_table


@dataclass
class MLAMetadata(BaseAttnMetadata):
    # fmt: off
    qo_indptr_cpu:     torch.Tensor  # on cpu
    kv_indptr_cpu:     torch.Tensor  # on cpu
    kv_indices:        torch.Tensor  # on gpu
    kv_len_arr_cpu:    torch.Tensor  # on cpu (query length of each request)
    num_qo_heads:      int
    kv_lora_rank:      int
    qk_rope_head_dim:  int
    page_size:         Literal[1]
    causal:            bool
    wrapper:           BatchMLAPagedAttentionWrapper
    initialized:       bool = False
    fast_plan:         bool = False
    # fmt: on

    def __post_init__(self) -> None:
        assert self.page_size == 1, "Currently only page_size=1 is supported."
        assert (
            self.qo_indptr_cpu.is_cpu
            and self.kv_indptr_cpu.is_cpu
            and self.kv_indices.is_cuda
            and self.kv_len_arr_cpu.is_cpu
        )

    def get_last_indices(self, bs: int) -> torch.Tensor:
        return self.qo_indptr_cpu[1 : 1 + bs] - 1


class MLABackend(BaseAttnBackend):
    """
    MLA Attention Backend using FlashInfer's BatchMLAPagedAttentionWrapper.

    This backend supports Multi-Latent Attention (MLA) which uses:
    - Compressed KV cache (ckv): kv_lora_rank dimension
    - Key position embedding (kpe): qk_rope_head_dim dimension
    """

    def __init__(self, config: ModelConfig, kvcache: MLAKVCache) -> None:
        from flashinfer.mla import BatchMLAPagedAttentionWrapper

        self.config = config
        self.kvcache = kvcache
        self.device = kvcache.device

        # MLA dimensions
        self.kv_lora_rank = kvcache.kv_lora_rank
        self.qk_rope_head_dim = kvcache.qk_rope_head_dim

        self.float_workspace_buffer = torch.empty(
            128 * 1024 * 1024, dtype=torch.uint8, device=self.device
        )

        # Use BatchMLAPagedAttentionWrapper for both prefill and decode
        self.prefill_wrapper = BatchMLAPagedAttentionWrapper(
            self.float_workspace_buffer,
            backend="auto",
        )
        self.decode_wrapper = BatchMLAPagedAttentionWrapper(
            self.float_workspace_buffer,
            backend="auto",
        )

        # Initialize some data members
        tp_size = get_tp_info().size
        self.qo_head_local = div_even(self.config.num_qo_heads, tp_size)

        self.cached_ones_cpu: torch.Tensor = torch.tensor([], dtype=torch.int32, pin_memory=True)

        # For CUDA graph
        self.capture_bs: List[int] = []
        self.max_graph_bs = 0
        self.graph_wrappers: Dict[int, BatchMLAPagedAttentionWrapper] = {}
        self.capture: MLACaptureData | None = None

    @staticmethod
    def _initialize_metadata_once(metadata: MLAMetadata) -> None:
        if metadata.initialized:
            return

        metadata.initialized = True
        sm_scale = 1.0 / math.sqrt(metadata.kv_lora_rank + metadata.qk_rope_head_dim)
        if metadata.fast_plan:
            fast_mla_decode_plan(
                metadata.wrapper,
                metadata.qo_indptr_cpu,
                metadata.kv_indptr_cpu,
                metadata.kv_indices,
                metadata.kv_len_arr_cpu,
                metadata.num_qo_heads,
                metadata.kv_lora_rank,
                metadata.qk_rope_head_dim,
                metadata.page_size,
                metadata.causal,
                sm_scale,
                torch.bfloat16,
                torch.bfloat16,
            )
        else:
            metadata.wrapper.plan(
                qo_indptr=metadata.qo_indptr_cpu,
                kv_indptr=metadata.kv_indptr_cpu,
                kv_indices=metadata.kv_indices,
                kv_len_arr=metadata.kv_len_arr_cpu,
                num_heads=metadata.num_qo_heads,
                head_dim_ckv=metadata.kv_lora_rank,
                head_dim_kpe=metadata.qk_rope_head_dim,
                page_size=metadata.page_size,
                causal=metadata.causal,
                sm_scale=sm_scale,
                q_data_type=torch.bfloat16,  # TODO: get from config
                kv_data_type=torch.bfloat16,
            )

    def _get_ones_cpu(self, bs: int) -> torch.Tensor:
        if bs <= len(self.cached_ones_cpu):
            return self.cached_ones_cpu[:bs]
        next_len = _next_power_of_2(bs)
        self.cached_ones_cpu = torch.ones(next_len, dtype=torch.int32, pin_memory=True)
        return self.cached_ones_cpu[:bs]

    def forward(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, layer_id: int, batch: Batch
    ) -> torch.Tensor:
        """
        MLA forward pass.

        Args:
            q: Query tensor [num_tokens, num_qo_heads * qk_head_dim]
               where qk_head_dim = kv_lora_rank + qk_rope_head_dim
            k: Latent representation [num_tokens, kv_lora_rank + qk_rope_head_dim]
               - k[..., :kv_lora_rank] = ckv (compressed KV)
               - k[..., kv_lora_rank:] = kpe (key position embedding)
            v: Ignored (V is extracted from K's ckv part)
            layer_id: Layer index
            batch: Batch info

        Returns:
            Attention output [num_tokens, num_qo_heads * kv_lora_rank]
        """
        metadata = batch.attn_metadata
        assert isinstance(metadata, MLAMetadata)
        self._initialize_metadata_once(metadata)

        # Store latent into cache
        self.kvcache.store_kv(k, v, batch.out_loc, layer_id)

        num_tokens = q.shape[0]

        # Reshape Q into [num_tokens, num_qo_heads, qk_head_dim]
        q = q.view(num_tokens, self.qo_head_local, -1)

        # Split Q into q_nope and q_pe
        # q_nope: [num_tokens, num_qo_heads, kv_lora_rank]
        # q_pe: [num_tokens, num_qo_heads, qk_rope_head_dim]
        q_nope = q[..., : self.kv_lora_rank]
        q_pe = q[..., self.kv_lora_rank :]

        # Get ckv and kpe from cache
        ckv_cache = self.kvcache.ckv_cache(layer_id)  # [cache_size, 1, kv_lora_rank]
        kpe_cache = self.kvcache.kpe_cache(layer_id)  # [cache_size, 1, qk_rope_head_dim]

        # Allocate output
        output = torch.empty(
            num_tokens, self.qo_head_local, self.kv_lora_rank,
            dtype=q.dtype, device=q.device
        )

        # Run MLA attention
        metadata.wrapper.run(
            q_nope=q_nope,
            q_pe=q_pe,
            ckv_cache=ckv_cache,
            kpe_cache=kpe_cache,
            out=output,
        )

        # Flatten output
        return output.view(num_tokens, self.qo_head_local * self.kv_lora_rank)

    def prepare_metadata(self, batch: Batch) -> None:
        reqs = batch.padded_reqs
        padded_size = len(reqs)
        seqlens_q = [req.extend_len for req in reqs]
        seqlens_k = [req.device_len for req in reqs]
        cached_lens = [req.cached_len for req in reqs]
        max_seqlen_q = max(seqlens_q)
        CPU_KWARGS = {"device": "cpu", "dtype": torch.int32, "pin_memory": True}

        # Build qo_indptr (query indptr)
        if max_seqlen_q == 1:  # decode with all extend_len = 1
            qo_indptr_cpu = torch.arange(0, padded_size + 1, **CPU_KWARGS)
        elif all(l == 0 for l in cached_lens):  # prefill with no cache hit
            qo_indptr_cpu = torch.tensor([0] + seqlens_q, **CPU_KWARGS).cumsum_(dim=0)
        else:  # normal extend prefill, with partial cache hit
            qo_indptr_cpu = torch.tensor([0] + seqlens_q, **CPU_KWARGS).cumsum_(dim=0)

        # Build kv_indptr and kv_indices
        kv_indptr_cpu = torch.tensor([0] + seqlens_k, **CPU_KWARGS).cumsum_(dim=0)
        if (
            self.capture is not None
            and padded_size in self.capture_bs
            and reqs
            and reqs[0].table_idx >= get_global_ctx().page_table.size(0) - self.max_graph_bs
        ):
            # Graph capture/replay dummy requests use synthetic table_idx values that
            # intentionally live beyond the scheduler-managed page table range. In
            # that case, kv indices must come from the static capture buffers rather
            # than from the global page table.
            page_table = self.capture.page_table[:padded_size, :]
            base_table_idx = reqs[0].table_idx
        else:
            page_table = get_global_ctx().page_table
            base_table_idx = 0
        kv_indices = torch.cat(
            [page_table[req.table_idx - base_table_idx, : req.device_len] for req in reqs]
        )

        # FlashInfer MLA expects the KV lengths for each request here.
        kv_len_arr_cpu = torch.tensor(seqlens_k, **CPU_KWARGS)

        batch.attn_metadata = MLAMetadata(
            qo_indptr_cpu=qo_indptr_cpu,
            kv_indptr_cpu=kv_indptr_cpu,
            kv_indices=kv_indices,
            kv_len_arr_cpu=kv_len_arr_cpu,
            num_qo_heads=self.qo_head_local,
            kv_lora_rank=self.kv_lora_rank,
            qk_rope_head_dim=self.qk_rope_head_dim,
            page_size=1,
            causal=batch.is_prefill,
            wrapper=self.decode_wrapper if batch.is_decode else self.prefill_wrapper,
        )


    def init_capture_graph(self, max_seq_len: int, bs_list: List[int]) -> None:
        from flashinfer.mla import BatchMLAPagedAttentionWrapper

        assert self.capture is None, "Capture already initialized."
        max_bs = max(bs_list)
        capture = MLACaptureData.create(max_bs, max_seq_len, self.kvcache.device)
        self.max_graph_bs = max_bs
        self.capture = capture
        self.capture_bs = sorted(bs_list)

        # FlashInfer MLA requires explicit replay buffers when CUDA graph is enabled.
        # Without these static buffers, the eager wrapper state is not replay-safe.
        for bs in bs_list:
            self.graph_wrappers[bs] = BatchMLAPagedAttentionWrapper(
                self.float_workspace_buffer,
                use_cuda_graph=True,
                qo_indptr=capture.cu_seqlens_q[: bs + 1],
                kv_indptr=capture.cu_seqlens_k[: bs + 1],
                kv_indices=capture.page_table[:bs, :].reshape(-1),
                kv_len_arr=capture.seq_lens[:bs],
                backend="auto",
            )

    @cached_property
    def use_tensor_cores(self) -> bool:
        if (overriden_value := ENV.FLASHINFER_USE_TENSOR_CORES.value) is not None:
            logger.warning(f"Overriding FlashInfer tensor core usage to {overriden_value}")
            return overriden_value
        return False  # MLA doesn't use tensor cores in the same way

    def prepare_for_capture(self, batch: Batch) -> None:
        bs = batch.size
        assert bs in self.capture_bs and bs in self.graph_wrappers and self.capture

        # Seed capture page table with the same dummy-page layout used by the main
        # engine graph dummy requests. page_size is fixed to 1 for MLA, so a dense
        # arange over max_seq_len is sufficient.
        capture = self.capture
        max_seq_len = capture.page_table.size(1)
        capture.page_table[:bs, :max_seq_len].copy_(
            torch.arange(max_seq_len, device=self.device, dtype=torch.int32).expand(bs, -1)
        )

        self.prepare_metadata(batch)
        metadata = batch.attn_metadata
        assert isinstance(metadata, MLAMetadata)
        metadata.wrapper = self.graph_wrappers[bs]
        self._initialize_metadata_once(metadata)

    def prepare_for_replay(self, batch: Batch) -> None:
        metadata, bs = batch.attn_metadata, batch.padded_size
        assert isinstance(metadata, MLAMetadata) and not metadata.initialized
        assert self.capture is not None and bs in self.capture_bs
        # CUDA graph replay must update the static buffers that were bound into the
        # graph wrapper at capture time. Without these copies, decode replay can use
        # stale dummy-request indices/lengths from graph capture, which is especially
        # visible on long-context MLA workloads.
        q_len = metadata.qo_indptr_cpu.size(0)
        kv_len = metadata.kv_indptr_cpu.size(0)
        self.capture.cu_seqlens_q[:q_len].copy_(
            metadata.qo_indptr_cpu.to(self.device, non_blocking=True)
        )
        self.capture.cu_seqlens_k[:kv_len].copy_(
            metadata.kv_indptr_cpu.to(self.device, non_blocking=True)
        )
        flat_len = metadata.kv_indices.numel()
        self.capture.page_table.view(-1)[:flat_len].copy_(metadata.kv_indices)
        self.capture.seq_lens[:bs].copy_(
            metadata.kv_len_arr_cpu.to(self.device, non_blocking=True)
        )
        # Replay-safe wrappers must read from the static buffers that were bound
        # during graph capture. Copying into capture tensors is insufficient if
        # metadata still points plan() at the per-request dynamic tensors.
        metadata.qo_indptr_cpu = self.capture.cu_seqlens_q[:q_len].cpu()
        metadata.kv_indptr_cpu = self.capture.cu_seqlens_k[:kv_len].cpu()
        metadata.kv_indices = self.capture.page_table[:bs, :].reshape(-1)[:flat_len]
        metadata.kv_len_arr_cpu = self.capture.seq_lens[:bs].cpu()
        metadata.wrapper = self.graph_wrappers[bs]
        metadata.fast_plan = False
        self._initialize_metadata_once(metadata)


def fast_mla_decode_plan(
    wrapper,
    qo_indptr_cpu: torch.Tensor,
    kv_indptr_cpu: torch.Tensor,
    kv_indices: torch.Tensor,
    kv_len_arr_cpu: torch.Tensor,
    num_heads: int,
    head_dim_ckv: int,
    head_dim_kpe: int,
    page_size: int,
    causal: bool,
    sm_scale: float,
    q_data_type: torch.dtype,
    kv_data_type: torch.dtype,
) -> None:
    """Fast replay-only MLA plan compatible with the local flashinfer build."""
    wrapper._causal = causal
    wrapper._page_size = page_size
    wrapper._sm_scale = sm_scale
    try:
        wrapper._cached_module.plan(
            wrapper._float_workspace_buffer,
            wrapper._int_workspace_buffer,
            wrapper._pin_memory_int_workspace_buffer,
            qo_indptr_cpu,
            kv_indptr_cpu,
            kv_len_arr_cpu,
            num_heads,
            head_dim_ckv,
            causal,
        )
    except TypeError:
        # Older flashinfer builds keep the legacy signature with plan_info only.
        wrapper._cached_module.plan(
            wrapper._float_workspace_buffer,
            wrapper._int_workspace_buffer,
            wrapper._plan_info,
            qo_indptr_cpu,
            kv_indptr_cpu,
            kv_len_arr_cpu,
            num_heads,
            head_dim_ckv,
            causal,
        )
