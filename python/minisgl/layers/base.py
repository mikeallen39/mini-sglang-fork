from __future__ import annotations

import re
from abc import abstractmethod
from typing import Any, Dict, Generic, List, TypeAlias, TypeVar

import torch

_STATE_DICT: TypeAlias = Dict[str, torch.Tensor]

_EXPERT_KEY_SUFFIXES = ("", ".weight", ".bias")


def _collect_expert_keys(
    state_dict: _STATE_DICT, prefix: str, param_name: str
) -> List[str]:
    """Collect expert weight keys in O(num_experts) via direct dict lookup."""
    keys: List[str] = []
    idx = 0
    while True:
        found = False
        for suffix in _EXPERT_KEY_SUFFIXES:
            candidate = f"{prefix}.{idx}.{param_name}{suffix}"
            if candidate in state_dict:
                keys.append(candidate)
                found = True
                break
        if not found:
            break
        idx += 1

    if keys:
        return keys

    # Fallback: linear scan for non-standard key naming conventions
    for key in list(state_dict.keys()):
        if prefix in key and param_name in key:
            keys.append(key)

    def _expert_index(k: str) -> int:
        match = re.search(r"experts\.(\d+)\.", k)
        return int(match.group(1)) if match else 0

    keys.sort(key=_expert_index)
    return keys


def _concat_prefix(prefix: str, name: str) -> str:
    return f"{prefix}.{name}" if prefix else name


class BaseOP:
    @abstractmethod
    def forward(self, *args: Any, **kwargs: Any) -> Any: ...

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.forward(*args, **kwargs)

    def _count_tensors(self) -> int:
        """Count total number of tensors in this module and submodules."""
        count = 0
        for name, param in self.__dict__.items():
            if name.startswith("_"):
                continue
            if isinstance(param, torch.Tensor):
                count += 1
            elif isinstance(param, BaseOP):
                count += param._count_tensors()
        return count

    def to(self, device: torch.device, pbar=None):
        """Move all tensor attributes to the specified device."""
        for name, param in self.__dict__.items():
            if name.startswith("_"):
                continue
            if isinstance(param, torch.Tensor):
                if param.is_meta:
                    raise RuntimeError(
                        f"Tensor '{name}' is still a meta tensor (no data loaded). "
                        f"Shape: {param.shape}, dtype: {param.dtype}. "
                        f"This indicates a weight loading issue."
                    )
                setattr(self, name, param.to(device))
                if pbar is not None:
                    pbar.update(1)
            elif isinstance(param, BaseOP):
                param.to(device, pbar=pbar)
        return self

    def named_children(self):
        """Iterate over all child modules."""
        for name, param in self.__dict__.items():
            if name.startswith("_"):
                continue
            if isinstance(param, BaseOP):
                yield name, param

    def named_parameters(self, prefix: str = ""):
        """Iterate over all parameters in the model."""
        for name, param in self.__dict__.items():
            if name.startswith("_"):
                continue
            if isinstance(param, torch.Tensor):
                yield _concat_prefix(prefix, name), param
            elif isinstance(param, BaseOP):
                yield from param.named_parameters(_concat_prefix(prefix, name))

    def state_dict(self, *, prefix: str = "", result: _STATE_DICT | None = None) -> _STATE_DICT:
        result = result if result is not None else {}

        for name, param in self.__dict__.items():
            if name.startswith("_"):
                continue
            if isinstance(param, torch.Tensor):
                result[_concat_prefix(prefix, name)] = param
            elif isinstance(param, BaseOP):
                param.state_dict(prefix=_concat_prefix(prefix, name), result=result)

        return result

    def load_state_dict(
        self,
        state_dict: _STATE_DICT,
        *,
        prefix: str = "",
        _internal: bool = False,
    ) -> None:
        for name, param in self.__dict__.items():
            if name.startswith("_"):
                continue

            if isinstance(param, torch.Tensor):
                if _concat_prefix(prefix, name) not in state_dict:
                    raise KeyError(f"Key '{_concat_prefix(prefix, name)}' not found in state_dict")
                item = state_dict.pop(_concat_prefix(prefix, name))

                assert isinstance(item, torch.Tensor)
                if param.shape != item.shape or param.dtype != item.dtype:
                    raise AssertionError(
                        f"State dict mismatch for {_concat_prefix(prefix, name)}: "
                        f"model shape={tuple(param.shape)} dtype={param.dtype}, "
                        f"state shape={tuple(item.shape)} dtype={item.dtype}"
                    )

                setattr(self, name, item)

            elif isinstance(param, BaseOP):
                param.load_state_dict(
                    state_dict, prefix=_concat_prefix(prefix, name), _internal=True
                )

        if not _internal and state_dict:
            raise RuntimeError(f"Unexpected keys in state_dict: {list(state_dict.keys())}")


class StateLessOP(BaseOP):
    def __init__(self):
        super().__init__()

    def load_state_dict(
        self,
        state_dict: _STATE_DICT,
        *,
        prefix: str = "",
        _internal: bool = False,
    ) -> None:
        if not _internal and state_dict:
            _ = prefix
            raise RuntimeError(f"Unexpected keys in state_dict: {list(state_dict.keys())}")

    def state_dict(self, *, prefix: str = "", result: _STATE_DICT | None = None) -> _STATE_DICT:
        _ = prefix
        return result if result is not None else {}


T = TypeVar("T", bound=BaseOP)


class OPList(BaseOP, Generic[T]):
    def __init__(self, ops: List[T]):
        super().__init__()
        self.op_list = ops

    def _count_tensors(self) -> int:
        """Count total number of tensors in all ops."""
        return sum(op._count_tensors() for op in self.op_list)

    def to(self, device: torch.device, pbar=None):
        """Move all ops to the specified device."""
        for i, op in enumerate(self.op_list):
            try:
                op.to(device, pbar=pbar)
            except RuntimeError as e:
                raise RuntimeError(f"In OPList[{i}]: {e}") from e
        return self

    def named_parameters(self, prefix: str = ""):
        """Iterate over all parameters in the OPList."""
        for i, op in enumerate(self.op_list):
            yield from op.named_parameters(_concat_prefix(prefix, str(i)))

    def state_dict(self, *, prefix: str = "", result: _STATE_DICT | None = None) -> _STATE_DICT:
        result = result if result is not None else {}
        for i, op in enumerate(self.op_list):
            op.state_dict(prefix=_concat_prefix(prefix, str(i)), result=result)
        return result

    def load_state_dict(
        self,
        state_dict: _STATE_DICT,
        *,
        prefix: str = "",
        _internal: bool = False,
    ) -> None:
        for i, op in enumerate(self.op_list):
            op.load_state_dict(state_dict, prefix=_concat_prefix(prefix, str(i)), _internal=True)

        if not _internal and state_dict:
            raise RuntimeError(f"Unexpected keys in state_dict: {list(state_dict.keys())}")
