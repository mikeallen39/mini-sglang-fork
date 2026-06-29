from __future__ import annotations
from typing import TYPE_CHECKING, List, NamedTuple, NoReturn, Set, Tuple, TypeAlias

import torch
import time
from minisgl.core import Batch, Req, SamplingParams
from minisgl.env import ENV
from minisgl.message import (
    AbortBackendMsg,
    ProfileBackendMsg,
    BaseBackendMsg,
    BatchBackendMsg,
    DetokenizeMsg,
    ExitMsg,
    UserMsg,
)
from minisgl.speculative import DFlashDraftInput, DFlashVerifyInput
from minisgl.utils import init_logger, load_tokenizer

from .cache import CacheManager
from .config import SchedulerConfig
from .decode import DecodeManager
from .io import SchedulerIOMixin
from .prefill import ChunkedReq, PrefillManager
from .table import TableManager

if TYPE_CHECKING:
    from minisgl.engine import BatchSamplingArgs, ForwardOutput


logger = init_logger(__name__)

Indice2D: TypeAlias = Tuple[torch.Tensor, torch.Tensor]


# For overlap scheduling, we also need to cache some other data to avoid IMA
class ForwardInput(NamedTuple):
    batch: Batch
    sample_args: BatchSamplingArgs
    input_tuple: Indice2D  # (token_mapping, positions)
    write_tuple: Indice2D  # (req_mapping, seq_lens or 0)


ForwardData: TypeAlias = "Tuple[ForwardInput, ForwardOutput]"


class Scheduler(SchedulerIOMixin):
    def __init__(self, config: SchedulerConfig):
        from minisgl.engine import Engine

        self.config = config
        self.engine = Engine(config)
        # Initialize the I/O mixin
        super().__init__(config, self.engine.world_cpu_group)

        # use another stream to overlap metadata processing with computation
        self.device = self.engine.device
        self.stream = torch.cuda.Stream(device=self.device)
        self.engine_stream_ctx = torch.cuda.stream(self.engine.stream)
        torch.cuda.set_stream(self.stream)

        # initialize other managers
        self.table_manager = TableManager(config.max_running_req, self.engine.page_table)
        self.cache_manager = CacheManager(
            self.device, self.engine.num_pages, config.page_size, config.cache_type
        )
        self.decode_manager = DecodeManager(config.page_size)
        self.prefill_manager = PrefillManager(
            self.cache_manager,
            self.table_manager,
            self.decode_manager,
            enable_prefix_cache=self.engine.model.supports_prefix_cache,
        )

        # some alias for easy access
        self.tp_info = config.tp_info
        self.world_info = config.world_info
        self.finished_reqs: Set[Req] = set()
        self.tokenizer = load_tokenizer(config.model_path)
        eos_token_id = self.tokenizer.eos_token_id
        if isinstance(eos_token_id, int):
            self.eos_token_ids = {eos_token_id}
        elif eos_token_id is None:
            self.eos_token_ids = set()
        else:
            self.eos_token_ids = set(eos_token_id)
        if config.model_config.model_type == "glm4_moe_lite":
            # GLM chat turns are delimited by role tokens rather than EOS.
            for token in ("<|user|>", "<|assistant|>", "<|observation|>"):
                token_id = self.tokenizer.convert_tokens_to_ids(token)
                if isinstance(token_id, int) and token_id >= 0:
                    self.eos_token_ids.add(token_id)
        self.page_table = self.engine.page_table
        self.page_size = config.page_size
        self.token_pool = self.table_manager.token_pool
        self.prefill_budget = config.max_extend_tokens
        self._decode_batch_buffers: list[dict[str, torch.Tensor]] = []
        self._decode_batch_buffer_index = 0
        self._dflash_profile_count = 0
        self._dflash_profile_draft_ms = 0.0
        self._dflash_profile_verify_ms = 0.0
        self._dflash_profile_update_ms = 0.0
        self._dflash_profile_accept_tokens = 0
        self._dflash_profile_total_tokens = 0
        self._run_startup_prewarm()

    def _process_last_data(self, last_data: ForwardData | None) -> None:
        if last_data is None:
            return

        batch, forward_output = last_data[0].batch, last_data[1]
        _, next_tokens_cpu, copy_done, hidden_states = forward_output
        copy_done.synchronize()
        reply: List[DetokenizeMsg] = []
        new_finished_reqs: Set[Req] = set()
        if (
            self.config.speculative_algorithm == "DFLASH"
            and batch.is_prefill
            and batch.mode == "target"
        ):
            self._update_dflash_prefill_state(batch, next_tokens_cpu, hidden_states)
        with self.cache_manager.lazy_free_region():
            for i, req in enumerate(batch.reqs):
                if isinstance(req, ChunkedReq):
                    continue
                next_token = next_tokens_cpu[i]
                req.append_host(next_token.unsqueeze(0))
                next_token = int(next_token.item())
                finished = not req.can_decode
                if not req.sampling_params.ignore_eos:
                    finished |= next_token in self.eos_token_ids
                reply.append(DetokenizeMsg(uid=req.uid, next_token=next_token, finished=finished))

                # NOTE: overlap scheduling may make the request freed twice, skip second free
                if finished and req not in self.finished_reqs:
                    self.decode_manager.remove_req(req)
                    self._free_req_resources(req)
                    new_finished_reqs.add(req)
        self.finished_reqs = new_finished_reqs
        self.send_result(reply)

    def _update_dflash_prefill_state(
        self,
        batch: Batch,
        next_tokens_cpu: torch.Tensor,
        hidden_states: torch.Tensor | None,
    ) -> None:
        if self.engine.draft_model is None:
            raise RuntimeError("DFLASH prefill state update requires a loaded draft model.")
        if hidden_states is None:
            raise RuntimeError(
                "DFLASH prefill expected target hidden capture, but got None. "
                "Target model capture wiring is incomplete."
            )
        if self.engine.dflash_mask_token_id is None:
            raise RuntimeError("DFLASH mask_token_id is not initialized.")
        if batch.prefill_extend_lens is None:
            raise RuntimeError(
                "DFLASH prefill state update requires prefill_extend_lens to be captured."
            )

        lengths = list(batch.prefill_extend_lens)
        offset = 0
        hidden_cpu = hidden_states.to(device="cpu")
        for i, req in enumerate(batch.reqs):
            length = lengths[i]
            req_hidden = hidden_cpu[offset : offset + length]
            offset += length
            req.speculative_target_hidden = req_hidden.contiguous()
            projected = self.engine.draft_model.project_target_hidden(
                req_hidden.to(device=self.device, dtype=self.engine.dtype)
            )
            req.speculative_draft_hidden = projected.to(device="cpu")
            req.speculative_verified_id = next_tokens_cpu[i : i + 1].clone()
            req.speculative_draft_tokens = None
        assert offset == hidden_states.shape[0]

    def _build_dflash_draft_block(self, req: Req) -> tuple[torch.Tensor, torch.Tensor]:
        if self.engine.draft_model is None:
            raise RuntimeError("DFLASH draft model is not loaded.")
        if req.speculative_verified_id is None:
            raise RuntimeError("DFLASH req is missing speculative_verified_id.")
        if req.speculative_draft_hidden is None:
            raise RuntimeError("DFLASH req is missing speculative_draft_hidden.")

        target_model = self.engine.model
        model_core = getattr(target_model, "model", None)
        if model_core is None or not hasattr(model_core, "embed_tokens"):
            raise RuntimeError("DFLASH target model is missing model.embed_tokens.")
        if not hasattr(target_model, "lm_head"):
            raise RuntimeError("DFLASH target model is missing lm_head.")

        verified_id = req.speculative_verified_id.to(device=self.device, dtype=torch.long)
        draft_hidden = req.speculative_draft_hidden.to(device=self.device, dtype=self.engine.dtype)
        prefix_lens = torch.tensor([req.cached_len], device=self.device, dtype=torch.long)
        return self.engine.draft_model.draft_block_greedy(
            verified_id=verified_id,
            draft_context=draft_hidden[-1:, :],
            prefix_lens=prefix_lens,
            target_embedding=model_core.embed_tokens,
            target_lm_head=target_model.lm_head,
            block_size=self.config.speculative_num_draft_tokens,
        )

    def _build_dflash_draft_input(self, req: Req) -> DFlashDraftInput:
        if req.speculative_verified_id is None:
            raise RuntimeError("DFLASH req is missing speculative_verified_id.")
        if req.speculative_target_hidden is None:
            raise RuntimeError("DFLASH req is missing speculative_target_hidden.")
        return DFlashDraftInput(
            verified_id=req.speculative_verified_id.to(device=self.device, dtype=torch.long),
            target_hidden=req.speculative_target_hidden.to(device=self.device, dtype=self.engine.dtype),
            ctx_lens=torch.tensor(
                [int(req.speculative_target_hidden.shape[0])],
                device=self.device,
                dtype=torch.int32,
            ),
            draft_seq_lens=torch.tensor([req.cached_len], device=self.device, dtype=torch.int32),
        )

    def _update_dflash_step_state(self, req: Req, token_id: int, hidden_states: torch.Tensor | None) -> None:
        req.speculative_verified_id = torch.tensor([token_id], dtype=torch.int32)
        if hidden_states is None:
            return
        hidden_cpu = hidden_states.to(device="cpu").contiguous()
        req.speculative_target_hidden = hidden_cpu
        projected = self.engine.draft_model.project_target_hidden(
            hidden_states.to(device=self.device, dtype=self.engine.dtype)
        )
        req.speculative_draft_hidden = projected.to(device="cpu")

    def _prepare_dflash_verify_batch(self, req: Req, verify_req: Req, verify_batch: Batch) -> None:
        verify_batch.padded_reqs = verify_batch.reqs
        verify_batch.prefill_extend_lens = [verify_req.extend_len]
        self.page_table[verify_req.table_idx].fill_(-1)
        self.page_table[verify_req.table_idx, : req.cached_len].copy_(
            self.page_table[req.table_idx, : req.cached_len]
        )
        self.token_pool[verify_req.table_idx, : verify_req.device_len].copy_(
            verify_req.input_ids.to(device=self.device, dtype=torch.int32)
        )
        self.cache_manager.allocate_paged(verify_batch.reqs, self.page_table)

    def _finalize_dflash_verify_batch(self, req: Req, verify_req: Req, verify_batch: Batch) -> None:
        _ = req, verify_batch
        self.cache_manager._free(self.page_table[verify_req.table_idx, verify_req.cached_len : verify_req.device_len])
        self.page_table[verify_req.table_idx].fill_(-1)

    def _append_req_token(self, req: Req, token_id: int) -> bool:
        req.cached_len = req.device_len
        req.device_len += 1
        req.append_host(torch.tensor([token_id], dtype=torch.int32))
        finished = not req.can_decode
        if not req.sampling_params.ignore_eos:
            finished |= token_id in self.eos_token_ids
        return finished

    def _run_one_target_decode_step(self, req: Req) -> tuple[int, bool]:
        batch = Batch(reqs=[req], phase="decode", mode="verify")
        batch.capture_hidden_layer_ids = self.engine.dflash_target_layer_ids
        forward_input = self._prepare_batch(batch)
        with self.engine.ctx.forward_batch(batch):
            logits = self.engine.model.forward()
            hidden_states = None
            getter = getattr(self.engine.model, "get_last_hidden_capture", None)
            if callable(getter):
                hidden_states = getter()
        sample_args = self.engine.sampler.prepare(batch)
        next_token_gpu = self.engine.sampler.sample(logits[: batch.size], sample_args).to(torch.int32)
        next_token = int(next_token_gpu[0].item())
        finished = self._append_req_token(req, next_token)
        self._update_dflash_step_state(req, next_token, hidden_states)
        return next_token, finished

    def _run_dflash_single_decode(self, req: Req) -> bool:
        if self.config.speculative_algorithm != "DFLASH":
            return False
        if self.engine.draft_model is None:
            return False
        if req.speculative_verified_id is None or req.speculative_draft_hidden is None:
            return False
        if req.speculative_target_hidden is None:
            return False
        if len(self.decode_manager.running_reqs) != 1:
            return False

        t0 = time.perf_counter()
        draft_input = self._build_dflash_draft_input(req)
        draft_tokens, draft_hidden = self._build_dflash_draft_block(req)
        req.speculative_target_hidden = draft_input.target_hidden.to(device="cpu")
        t1 = time.perf_counter()
        verify_input = DFlashVerifyInput(
            draft_token=draft_tokens.to(device=self.device, dtype=torch.long),
            draft_token_num=max(int(draft_tokens.shape[1]) - 1, 0),
        )
        accept_len, bonus, next_hidden = self.engine.forward_verify_greedy(
            batch=Batch(reqs=[req], phase="decode", mode="verify"),
            verify_input=verify_input,
            prepare_metadata=self.engine.attn_backend.prepare_metadata,
            make_positions=_make_positions,
            make_input_tuple=_make_input_tuple,
            token_pool=self.token_pool,
            prepare_verify_batch=self._prepare_dflash_verify_batch,
            finalize_verify_batch=self._finalize_dflash_verify_batch,
        )
        t2 = time.perf_counter()
        accept_len_i = int(accept_len[0].item())
        bonus_token = int(bonus[0].item())
        reply: List[DetokenizeMsg] = []
        with self.cache_manager.lazy_free_region():
            finished = False
            accepted_tokens = draft_tokens[0, 1 : 1 + accept_len_i].tolist()
            for token in accepted_tokens:
                finished = self._append_req_token(req, int(token))
                reply.append(DetokenizeMsg(uid=req.uid, next_token=int(token), finished=False))
                if finished:
                    break
            if not finished:
                finished = self._append_req_token(req, bonus_token)
                reply.append(DetokenizeMsg(uid=req.uid, next_token=bonus_token, finished=False))

            if reply:
                reply[-1].finished = finished
            if finished:
                self.decode_manager.remove_req(req)
                self._free_req_resources(req)
                self.finished_reqs = {req}
            else:
                self.finished_reqs = set()

        if not finished:
            self._update_dflash_step_state(req, bonus_token, next_hidden)
        t3 = time.perf_counter()
        req.speculative_draft_tokens = verify_input.draft_token.to(device="cpu", dtype=torch.int32)
        req.speculative_accept_len = accept_len_i
        if draft_hidden is not None:
            req.speculative_draft_hidden = draft_hidden[0].to(device="cpu")
        self._dflash_profile_count += 1
        self._dflash_profile_draft_ms += (t1 - t0) * 1000.0
        self._dflash_profile_verify_ms += (t2 - t1) * 1000.0
        self._dflash_profile_update_ms += (t3 - t2) * 1000.0
        self._dflash_profile_accept_tokens += accept_len_i
        self._dflash_profile_total_tokens += max(int(draft_tokens.shape[1]) - 1, 0)
        denom = max(self._dflash_profile_total_tokens, 1)
        logger.info_rank0(
            "DFLASH step %d: draft_ms=%.3f verify_ms=%.3f update_ms=%.3f "
            "accept_len=%d block=%d accept_rate_running=%.3f (%d/%d)",
            self._dflash_profile_count,
            (t1 - t0) * 1000.0,
            (t2 - t1) * 1000.0,
            (t3 - t2) * 1000.0,
            accept_len_i,
            max(int(draft_tokens.shape[1]) - 1, 0),
            self._dflash_profile_accept_tokens / denom,
            self._dflash_profile_accept_tokens,
            self._dflash_profile_total_tokens,
        )
        if self._dflash_profile_count % 10 == 0:
            logger.info_rank0(
                "DFLASH profile avg over %d steps: draft_ms=%.3f verify_ms=%.3f update_ms=%.3f "
                "accept_rate=%.3f (%d/%d)",
                self._dflash_profile_count,
                self._dflash_profile_draft_ms / self._dflash_profile_count,
                self._dflash_profile_verify_ms / self._dflash_profile_count,
                self._dflash_profile_update_ms / self._dflash_profile_count,
                self._dflash_profile_accept_tokens / denom,
                self._dflash_profile_accept_tokens,
                self._dflash_profile_total_tokens,
            )
        self.send_result(reply)
        return True
    def _handle_profile_msg(self, msg: ProfileBackendMsg) -> None:
        """Handle profile start/stop commands from the HTTP API."""
        import os
        if msg.action == "start":
            logger.info("Starting PyTorch profiler...")
            activities = [torch.profiler.ProfilerActivity.CPU]
            if torch.cuda.is_available():
                activities.append(torch.profiler.ProfilerActivity.CUDA)
            output_dir = msg.output_dir or os.environ.get("SGLANG_TORCH_PROFILER_DIR", "/tmp/minisgl_traces")
            os.makedirs(output_dir, exist_ok=True)
            self._profiler = torch.profiler.profile(
                activities=activities,
                record_shapes=False,
                with_stack=False,
            )
            self._profiler.__enter__()
            self._profile_step_count = 0
            self._profile_max_steps = msg.num_steps if msg.num_steps > 0 else None
            logger.info(f"Profiler started. output_dir={output_dir}, max_steps={self._profile_max_steps}")
        elif msg.action == "stop":
            if hasattr(self, "_profiler") and self._profiler is not None:
                logger.info("Stopping PyTorch profiler...")
                try:
                    self._profiler.__exit__(None, None, None)
                    torch.cuda.synchronize()
                    output_dir = msg.output_dir or os.environ.get("SGLANG_TORCH_PROFILER_DIR", "/tmp/minisgl_traces")
                    import time
                    trace_path = os.path.join(output_dir, f"{time.time()}-TP-0.trace.json")
                    self._profiler.export_chrome_trace(trace_path)
                    logger.info(f"Profile trace saved to {trace_path}")
                except Exception as e:
                    logger.error(f"Failed to stop profiler: {e}", exc_info=True)
                finally:
                    self._profiler = None


    def _check_profile_step(self) -> None:
        """Increment profile step counter and auto-stop if num_steps reached."""
        if not hasattr(self, "_profiler") or self._profiler is None:
            return
        self._profile_step_count += 1
        if self._profile_max_steps is not None and self._profile_step_count >= self._profile_max_steps:
            import os, time
            logger.info(f"Auto-stopping profiler after {self._profile_step_count} steps")
            self._profiler.__exit__(None, None, None)
            output_dir = os.environ.get("SGLANG_TORCH_PROFILER_DIR", "/tmp/minisgl_traces")
            os.makedirs(output_dir, exist_ok=True)
            trace_path = os.path.join(output_dir, f"{time.time()}-TP-0.trace.json")
            self._profiler.export_chrome_trace(trace_path)
            logger.info(f"Profile trace saved to {trace_path}")
            self._profiler = None

    def _process_one_msg(self, msg: BaseBackendMsg) -> None:
        if isinstance(msg, BatchBackendMsg):
            for msg in msg.data:
                self._process_one_msg(msg)
        elif isinstance(msg, ExitMsg):
            raise KeyboardInterrupt
        elif isinstance(msg, UserMsg):
            logger.debug_rank0("Received user msg: %s", msg)
            input_len, max_seq_len = len(msg.input_ids), self.engine.max_seq_len
            max_output_len = max_seq_len - input_len
            if max_output_len <= 0:
                return logger.warning_rank0(
                    f"Input sequence length {input_len} exceeds {max_seq_len}, "
                    f"request {msg.uid} is dropped."
                )
            if msg.sampling_params.max_tokens > max_output_len:
                msg.sampling_params.max_tokens = max_output_len
                logger.warning_rank0(
                    f"Adjust max_tokens to {max_output_len} for request {msg.uid}."
                )
            self.prefill_manager.add_one_req(msg)
        elif isinstance(msg, AbortBackendMsg):
            logger.debug_rank0("Aborting request %d", msg.uid)
            req_to_free = self.prefill_manager.abort_req(msg.uid)
            req_to_free = req_to_free or self.decode_manager.abort_req(msg.uid)
            if req_to_free is not None:
                self._free_req_resources(req_to_free)
        elif isinstance(msg, ProfileBackendMsg):
            self._handle_profile_msg(msg)
        else:
            logger.error(f"Unknown message type: {type(msg)}")
            raise NotImplementedError
            raise NotImplementedError

    def _free_req_resources(self, req: Req) -> None:
        self.engine.model.clear_runtime_state_slot(req.table_idx)
        self.table_manager.free(req.table_idx)
        self.cache_manager.free_and_cache_finished_req(req, self.page_table)

    def _prepare_batch(self, batch: Batch) -> ForwardInput:
        self.engine.graph_runner.pad_batch(batch)
        self.cache_manager.allocate_paged(batch.reqs, self.page_table)
        batch.prefill_extend_lens = [req.extend_len for req in batch.reqs] if batch.is_prefill else None
        if (
            self.config.speculative_algorithm == "DFLASH"
            and batch.is_prefill
            and self.engine.dflash_target_layer_ids is not None
        ):
            batch.capture_hidden_layer_ids = self.engine.dflash_target_layer_ids
        else:
            batch.capture_hidden_layer_ids = None
        mm_reqs = [req for req in batch.reqs if req.pixel_values is not None]
        if mm_reqs:
            if len(mm_reqs) != len(batch.reqs):
                raise NotImplementedError(
                    "Mixed multimodal and text-only batches are not supported yet."
                )
            if batch.is_decode:
                # Visual side inputs are only needed for the first prefill pass.
                batch.pixel_values = None
                batch.image_grid_thw = None
            else:
                if len(mm_reqs) != 1:
                    raise NotImplementedError(
                        "Only single-request multimodal prefill is supported at the moment."
                    )
                batch.pixel_values = mm_reqs[0].pixel_values
                batch.image_grid_thw = mm_reqs[0].image_grid_thw
                batch.mm_token_type_ids = mm_reqs[0].mm_token_type_ids
        else:
            batch.pixel_values = None
            batch.image_grid_thw = None
            batch.mm_token_type_ids = None
        if ENV.DECODE_BATCH_REUSE_BUFFERS.value and batch.is_decode:
            batch.text_positions, input_mapping, write_mapping = self._prepare_decode_batch_fast(batch)
        else:
            batch.text_positions = _make_positions(batch, self.device)
            batch.positions = batch.text_positions
            input_mapping = _make_input_tuple(batch, self.device)
            write_mapping = _make_write_tuple(batch, self.device)
        batch.positions = batch.text_positions
        batch.mrope_positions = None
        batch.out_loc = self.page_table[input_mapping]
        self.engine.attn_backend.prepare_metadata(batch)
        return ForwardInput(
            batch=batch,
            sample_args=self.engine.sampler.prepare(batch),
            input_tuple=input_mapping,
            write_tuple=write_mapping,
        )

    def _allocate_decode_batch_buffer_slot(self, max_reqs: int) -> dict[str, torch.Tensor]:
        return {
            "host_positions": torch.empty(max_reqs, dtype=torch.int32, pin_memory=True),
            "host_req_map": torch.empty(max_reqs, dtype=torch.int64, pin_memory=True),
            "host_write_map": torch.empty(max_reqs, dtype=torch.int64, pin_memory=True),
            "host_write_pos": torch.empty(max_reqs, dtype=torch.int64, pin_memory=True),
            "device_positions": torch.empty(max_reqs, dtype=torch.int32, device=self.device),
            "device_req_map": torch.empty(max_reqs, dtype=torch.int64, device=self.device),
            "device_write_map": torch.empty(max_reqs, dtype=torch.int64, device=self.device),
            "device_write_pos": torch.empty(max_reqs, dtype=torch.int64, device=self.device),
        }

    def _ensure_decode_batch_buffers(self, max_reqs: int) -> dict[str, torch.Tensor]:
        if not self._decode_batch_buffers:
            self._decode_batch_buffers = [
                self._allocate_decode_batch_buffer_slot(max_reqs),
                self._allocate_decode_batch_buffer_slot(max_reqs),
            ]
            self._decode_batch_buffer_index = 0
        else:
            for i, buffers in enumerate(self._decode_batch_buffers):
                if int(buffers["host_positions"].numel()) < max_reqs:
                    self._decode_batch_buffers[i] = self._allocate_decode_batch_buffer_slot(max_reqs)

        buffers = self._decode_batch_buffers[self._decode_batch_buffer_index]
        self._decode_batch_buffer_index = (self._decode_batch_buffer_index + 1) % len(
            self._decode_batch_buffers
        )
        return buffers

    def _prepare_decode_batch_fast(self, batch: Batch) -> tuple[torch.Tensor, Indice2D, Indice2D]:
        padded_size = len(batch.padded_reqs)
        req_size = len(batch.reqs)
        buffers = self._ensure_decode_batch_buffers(max(padded_size, req_size))

        host_positions = buffers["host_positions"][:padded_size]
        host_req_map = buffers["host_req_map"][:padded_size]
        host_write_map = buffers["host_write_map"][:req_size]
        host_write_pos = buffers["host_write_pos"][:req_size]

        for i, req in enumerate(batch.padded_reqs):
            host_positions[i] = req.cached_len
            host_req_map[i] = req.table_idx
        for i, req in enumerate(batch.reqs):
            host_write_map[i] = req.table_idx
            host_write_pos[i] = req.device_len if req.can_decode else -1

        device_positions = buffers["device_positions"][:padded_size]
        device_req_map = buffers["device_req_map"][:padded_size]
        device_write_map = buffers["device_write_map"][:req_size]
        device_write_pos = buffers["device_write_pos"][:req_size]

        device_positions.copy_(host_positions, non_blocking=True)
        device_req_map.copy_(host_req_map, non_blocking=True)
        device_write_map.copy_(host_write_map, non_blocking=True)
        device_write_pos.copy_(host_write_pos, non_blocking=True)

        positions = device_positions
        input_mapping: Indice2D = (device_req_map, positions.to(torch.int64))
        write_mapping: Indice2D = (device_write_map, device_write_pos)
        return positions, input_mapping, write_mapping

    def _schedule_next_batch(self) -> ForwardInput | None:
        # TODO: support other policies: e.g. DECODE first
        batch = (
            self.prefill_manager.schedule_next_batch(self.prefill_budget)
            or self.decode_manager.schedule_next_batch()
        )
        return self._prepare_batch(batch) if batch else None

    def _forward(self, forward_input: ForwardInput) -> ForwardOutput:
        batch, sample_args, input_mapping, output_mapping = forward_input
        batch.input_ids = self.token_pool[input_mapping]
        if ENV.OVERLAP_EXTRA_SYNC:  # NOTE: https://github.com/sgl-project/mini-sglang/issues/58
            self.stream.synchronize()
        forward_output = self.engine.forward_batch(batch, sample_args)
        self.token_pool[output_mapping] = forward_output.next_tokens_gpu
        self.decode_manager.filter_reqs(forward_input.batch.reqs)
        return forward_output

    def run_when_idle(self) -> None:
        """Called when the scheduler is idle to perform background tasks."""
        logger.info_rank0("Scheduler is idle, waiting for new reqs...")
        self.cache_manager.check_integrity()

    def overlap_loop(self, last_data: ForwardData | None) -> ForwardData | None:
        """
        The main loop of overlapping scheduling and execution.

        It will overlap the execution of current batch and processing of last batch's results,
        which can effectively hide CPU latency and improve GPU utilization.
        """
        blocking = not (
            last_data is not None  # don't block if we have a batch to be processed
            or self.prefill_manager.runnable
            or self.decode_manager.runnable
        )
        for msg in self.receive_msg(blocking=blocking):
            self._process_one_msg(msg)

        forward_input = self._schedule_next_batch()
        ongoing_data = None
        if forward_input is not None:
            with self.engine_stream_ctx:  # run the batch in the engine's stream
                self.engine.stream.wait_stream(self.stream)
                ongoing_data = (forward_input, self._forward(forward_input))

        self._process_last_data(last_data)
        self._check_profile_step()
        return ongoing_data

    def normal_loop(self) -> None:
        blocking = not (self.prefill_manager.runnable or self.decode_manager.runnable)
        for msg in self.receive_msg(blocking=blocking):
            self._process_one_msg(msg)

        if (
            self.config.speculative_algorithm == "DFLASH"
            and len(self.decode_manager.running_reqs) == 1
            and not self.prefill_manager.runnable
        ):
            only_req = next(iter(self.decode_manager.running_reqs))
            if self._run_dflash_single_decode(only_req):
                return

        forward_input = self._schedule_next_batch()
        ongoing_data = None
        if forward_input is not None:
            ongoing_data = (forward_input, self._forward(forward_input))

        self._process_last_data(ongoing_data)
        self._check_profile_step()

    @torch.inference_mode()
    def run_forever(self) -> NoReturn:
        if ENV.DISABLE_OVERLAP_SCHEDULING:
            with self.engine_stream_ctx:
                self.engine.stream.wait_stream(self.stream)
                while True:
                    self.normal_loop()
        else:
            assert torch.cuda.current_stream() == self.stream
            data = None
            while True:
                data = self.overlap_loop(data)

    def shutdown(self) -> None:
        torch.cuda.synchronize(self.device)
        self.sync_all_ranks()
        self.engine.shutdown()

    def _run_startup_prewarm(self) -> None:
        if not self.tp_info.is_primary() or self.tp_info.size != 1:
            return
        if self.config.quantization != "w8a8_int8_moe_only":
            return
        if self.config.moe_backend != "fused":
            return
        if self.config.linear_attn_backend != "sglang":
            return

        prewarm_lengths = [64, 256]
        logger.info_rank0(f"Start Triton prewarm for prefill lengths: {prewarm_lengths}")
        with self.engine_stream_ctx:
            self.engine.stream.wait_stream(self.stream)
            for length in prewarm_lengths:
                try:
                    self._run_prefill_prewarm_once(length)
                except NotImplementedError as exc:
                    logger.warning_rank0(
                        "Skip Triton prewarm for input_len=%d because cache manager cannot evict: %s",
                        length,
                        exc,
                    )
                except RuntimeError as exc:
                    logger.warning_rank0(
                        "Skip Triton prewarm for input_len=%d due to runtime error: %s",
                        length,
                        exc,
                    )
            torch.cuda.synchronize(self.device)
        logger.info_rank0("Triton prewarm finished.")

    def _run_prefill_prewarm_once(self, input_len: int) -> None:
        table_idx = self.table_manager.allocate()
        cache_handle = self.cache_manager.manager.match_prefix(
            torch.empty(0, dtype=torch.int32)
        )[0]
        self.cache_manager.lock(cache_handle)
        sampling_params = SamplingParams(max_tokens=1)
        req = Req(
            input_ids=torch.zeros(input_len, dtype=torch.int32),
            table_idx=table_idx,
            cached_len=0,
            output_len=1,
            uid=-1000 - input_len,
            sampling_params=sampling_params,
            cache_handle=cache_handle,
        )
        self.token_pool[table_idx][:input_len].zero_()

        try:
            batch = Batch(reqs=[req], phase="prefill")
            forward_input = self._prepare_batch(batch)
            _, next_tokens_cpu, copy_done = self._forward(forward_input)
            copy_done.synchronize()
            del next_tokens_cpu
        finally:
            with self.cache_manager.lazy_free_region():
                self._free_req_resources(req)


def _make_positions(batch: Batch, device: torch.device) -> torch.Tensor:
    needed_size = sum(r.extend_len for r in batch.padded_reqs)
    indices_host = torch.empty(needed_size, dtype=torch.int32, pin_memory=True)
    offset = 0
    for req in batch.padded_reqs:
        length = req.extend_len
        torch.arange(
            req.cached_len,
            req.device_len,
            dtype=torch.int32,
            out=indices_host[offset : offset + length],
        )
        offset += length
    return indices_host.to(device, non_blocking=True)


def _make_input_tuple(batch: Batch, device: torch.device) -> Indice2D:
    mapping_host = torch.empty(len(batch.positions), dtype=torch.int64, pin_memory=True)
    offset = 0
    for req in batch.padded_reqs:
        length = req.extend_len
        mapping_host[offset : offset + length].fill_(req.table_idx)
        offset += length
    return mapping_host.to(device, non_blocking=True), batch.positions.to(torch.int64)


def _make_write_tuple(batch: Batch, device: torch.device) -> Indice2D:
    mapping_list = [req.table_idx for req in batch.reqs]
    mapping_host = torch.tensor(mapping_list, dtype=torch.int64, pin_memory=True)
    write_list = [(req.device_len if req.can_decode else -1) for req in batch.reqs]
    write_host = torch.tensor(write_list, dtype=torch.int64, pin_memory=True)
    return mapping_host.to(device, non_blocking=True), write_host.to(device, non_blocking=True)
