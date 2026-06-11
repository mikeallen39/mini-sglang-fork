from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, List, Literal

import torch
import torch.distributed as dist

if TYPE_CHECKING:
    from minisgl.distributed import DistributedInfo
    from minisgl.kernel import PyNCCLCommunicator


@dataclass
class DistributedImpl(ABC):
    @abstractmethod
    def all_reduce(self, x: torch.Tensor) -> torch.Tensor: ...

    @abstractmethod
    def all_gather(self, x: torch.Tensor) -> torch.Tensor: ...


@dataclass
class TorchDistributedImpl(DistributedImpl):
    group: torch.distributed.ProcessGroup | None = None

    def all_reduce(self, x: torch.Tensor) -> torch.Tensor:
        tp_size = dist.get_world_size(group=self.group)
        if tp_size == 1:
            return x
        dist.all_reduce(x, op=dist.ReduceOp.SUM, group=self.group)
        return x

    def all_gather(self, x: torch.Tensor) -> torch.Tensor:
        tp_size = dist.get_world_size(group=self.group)
        if tp_size == 1:
            return x
        shape = list(x.shape)
        shape[0] = shape[0] * tp_size
        out = torch.empty(shape, dtype=x.dtype, device=x.device)
        dist.all_gather_into_tensor(out, x, group=self.group)
        return out


@dataclass
class PyNCCLDistributedImpl(DistributedImpl):
    comm: PyNCCLCommunicator
    size: int

    def all_reduce(self, x: torch.Tensor) -> torch.Tensor:
        self.comm.all_reduce(x, "sum")
        return x

    def all_gather(self, x: torch.Tensor) -> torch.Tensor:
        output_shape = list(x.shape)
        output_shape[0] *= self.size
        result = x.new_empty(output_shape)
        self.comm.all_gather(result, x)
        return result


class DistributedCommunicator:
    plugins: Dict[str, List[DistributedImpl]] = {
        "tp": [TorchDistributedImpl()],
        "world": [TorchDistributedImpl()],
    }

    def __init__(self, kind: Literal["tp", "world"] = "tp"):
        self.kind = kind

    def all_reduce(self, x: torch.Tensor) -> torch.Tensor:
        return self.plugins[self.kind][-1].all_reduce(x)

    def all_gather(self, x: torch.Tensor) -> torch.Tensor:
        return self.plugins[self.kind][-1].all_gather(x)


def configure_torch_distributed(
    *,
    tp_group: torch.distributed.ProcessGroup | None,
    world_group: torch.distributed.ProcessGroup | None,
) -> None:
    DistributedCommunicator.plugins = {
        "tp": [TorchDistributedImpl(tp_group)],
        "world": [TorchDistributedImpl(world_group)],
    }


def enable_pynccl_distributed(
    tp_info: DistributedInfo,
    tp_cpu_group: torch.distributed.ProcessGroup,
    max_bytes: int,
    *,
    world_info: DistributedInfo | None = None,
    world_cpu_group: torch.distributed.ProcessGroup | None = None,
) -> None:
    """
    Enable PyNCCL-based distributed communication for tensor parallelism.
    """
    from minisgl.kernel import init_pynccl

    if tp_info.size > 1:
        comm = init_pynccl(
            tp_rank=tp_info.rank,
            tp_size=tp_info.size,
            tp_cpu_group=tp_cpu_group,
            max_size_bytes=max_bytes,
        )
        DistributedCommunicator.plugins["tp"].append(
            PyNCCLDistributedImpl(comm, size=tp_info.size)
        )

    world_info = tp_info if world_info is None else world_info
    world_cpu_group = tp_cpu_group if world_cpu_group is None else world_cpu_group
    if world_info.size > 1:
        comm = init_pynccl(
            tp_rank=world_info.rank,
            tp_size=world_info.size,
            tp_cpu_group=world_cpu_group,
            max_size_bytes=max_bytes,
        )
        DistributedCommunicator.plugins["world"].append(
            PyNCCLDistributedImpl(comm, size=world_info.size)
        )


def destroy_distributed() -> None:
    """
    Destroy all the distributed communication plugins.
    """
    DistributedCommunicator.plugins = {
        "tp": [TorchDistributedImpl()],
        "world": [TorchDistributedImpl()],
    }
