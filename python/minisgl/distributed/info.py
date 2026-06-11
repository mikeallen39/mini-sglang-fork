from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DistributedInfo:  # should not export from here
    rank: int
    size: int

    def __post_init__(self):
        assert 0 <= self.rank < self.size

    def is_primary(self) -> bool:
        return self.rank == 0


_TP_INFO: DistributedInfo | None = None
_EP_INFO: DistributedInfo | None = None
_WORLD_INFO: DistributedInfo | None = None


def set_tp_info(rank: int, size: int) -> None:
    global _TP_INFO
    if _TP_INFO is not None:
        raise RuntimeError("TP info has been set")
    _TP_INFO = DistributedInfo(rank, size)


def set_ep_info(rank: int, size: int) -> None:
    global _EP_INFO
    if _EP_INFO is not None:
        raise RuntimeError("EP info has been set")
    _EP_INFO = DistributedInfo(rank, size)


def set_world_info(rank: int, size: int) -> None:
    global _WORLD_INFO
    if _WORLD_INFO is not None:
        raise RuntimeError("World info has been set")
    _WORLD_INFO = DistributedInfo(rank, size)


def get_tp_info() -> DistributedInfo:
    if _TP_INFO is None:
        raise RuntimeError("TP info has not been set")
    return _TP_INFO


def get_ep_info() -> DistributedInfo:
    if _EP_INFO is None:
        raise RuntimeError("EP info has not been set")
    return _EP_INFO


def get_world_info() -> DistributedInfo:
    if _WORLD_INFO is None:
        raise RuntimeError("World info has not been set")
    return _WORLD_INFO


def try_get_tp_info() -> DistributedInfo | None:
    return _TP_INFO


def try_get_ep_info() -> DistributedInfo | None:
    return _EP_INFO


def try_get_world_info() -> DistributedInfo | None:
    return _WORLD_INFO


def build_parallel_infos(
    world_rank: int,
    tp_size: int,
    ep_size: int,
) -> tuple[DistributedInfo, DistributedInfo, DistributedInfo]:
    if tp_size <= 0:
        raise ValueError(f"tp_size must be positive, got {tp_size}")
    if ep_size <= 0:
        raise ValueError(f"ep_size must be positive, got {ep_size}")
    world_size = tp_size * ep_size
    if not 0 <= world_rank < world_size:
        raise ValueError(
            f"world_rank must be in [0, {world_size}), got world_rank={world_rank}"
        )
    tp_rank = world_rank % tp_size
    ep_rank = world_rank // tp_size
    return (
        DistributedInfo(rank=world_rank, size=world_size),
        DistributedInfo(rank=tp_rank, size=tp_size),
        DistributedInfo(rank=ep_rank, size=ep_size),
    )


def build_ep_info(tp_rank: int, tp_size: int, ep_size: int) -> DistributedInfo:
    _, _, ep_info = build_parallel_infos(tp_rank, tp_size, ep_size)
    return ep_info


def get_local_expert_range(
    num_experts: int,
    ep_info: DistributedInfo | None = None,
) -> tuple[int, int]:
    ep_info = get_ep_info() if ep_info is None else ep_info
    if num_experts % ep_info.size != 0:
        raise ValueError(
            f"Number of experts ({num_experts}) must be divisible by ep_size ({ep_info.size})"
        )
    num_local_experts = num_experts // ep_info.size
    start = ep_info.rank * num_local_experts
    end = start + num_local_experts
    return start, end


def get_moe_tp_info(
    tp_info: DistributedInfo | None = None,
    ep_info: DistributedInfo | None = None,
) -> DistributedInfo:
    tp_info = get_tp_info() if tp_info is None else tp_info
    _ = get_ep_info() if ep_info is None else ep_info
    return tp_info


__all__ = [
    "DistributedInfo",
    "set_tp_info",
    "set_ep_info",
    "set_world_info",
    "get_tp_info",
    "get_ep_info",
    "get_world_info",
    "try_get_tp_info",
    "try_get_ep_info",
    "try_get_world_info",
    "build_parallel_infos",
    "build_ep_info",
    "get_local_expert_range",
    "get_moe_tp_info",
]
