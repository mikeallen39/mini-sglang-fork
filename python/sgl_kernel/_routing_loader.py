from __future__ import annotations

import importlib.metadata
import threading
from pathlib import Path

import torch

_LOAD_LOCK = threading.Lock()
_LOADED = False


def _candidate_common_ops_paths() -> list[Path]:
    candidates: list[Path] = []

    try:
        dist = importlib.metadata.distribution("sgl-kernel")
        base = Path(dist.locate_file(""))
        candidates.extend(
            [
                base / "sgl_kernel" / "sm80" / "common_ops.abi3.so",
                base / "sgl_kernel" / "sm90" / "common_ops.abi3.so",
                base / "sgl_kernel" / "sm100" / "common_ops.abi3.so",
            ]
        )
    except importlib.metadata.PackageNotFoundError:
        pass

    pkg_dir = Path(__file__).resolve().parent
    candidates.extend(pkg_dir.glob("sm*/common_ops.abi3.so"))

    deduped: list[Path] = []
    seen: set[Path] = set()
    for path in candidates:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        deduped.append(resolved)
    return deduped


def _resolve_routing_only_common_ops() -> Path:
    for candidate in _candidate_common_ops_paths():
        if candidate.exists():
            return candidate
    searched = "\n".join(str(path) for path in _candidate_common_ops_paths())
    raise FileNotFoundError(f"Could not find sgl_kernel common_ops.abi3.so. Searched:\n{searched}")


def ensure_routing_ops_loaded() -> None:
    global _LOADED
    if _LOADED:
        return
    with _LOAD_LOCK:
        if _LOADED:
            return
        torch.ops.load_library(str(_resolve_routing_only_common_ops()))
        _LOADED = True
