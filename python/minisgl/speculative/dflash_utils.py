from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
from typing import Any, List, Optional

import torch

DEFAULT_DFLASH_MASK_TOKEN = "<|MASK|>"


def build_target_layer_ids(num_target_layers: int, num_draft_layers: int) -> List[int]:
    if num_target_layers <= 0:
        raise ValueError(
            f"num_target_layers must be positive, got {num_target_layers}."
        )
    if num_draft_layers <= 0:
        raise ValueError(f"num_draft_layers must be positive, got {num_draft_layers}.")
    if num_draft_layers == 1:
        return [num_target_layers // 2]

    start = 1
    end = num_target_layers - 3
    if end < start:
        raise ValueError(
            "DFlash layer selection requires num_target_layers >= 4. "
            f"Got num_target_layers={num_target_layers}."
        )

    span = end - start
    return [
        int(round(start + (i * span) / (num_draft_layers - 1)))
        for i in range(num_draft_layers)
    ]


def _cfg_get(config: Any, key: str, default: Any = None) -> Any:
    if isinstance(config, dict):
        return config.get(key, default)
    return getattr(config, key, default)


def _get_text_config(config: Any) -> Any:
    if config is None:
        return None
    if isinstance(config, dict):
        return config.get("text_config", config)
    text_config = getattr(config, "text_config", None)
    if text_config is not None:
        return text_config
    get_text_config = getattr(config, "get_text_config", None)
    if callable(get_text_config):
        try:
            resolved = get_text_config()
            if resolved is not None:
                return resolved
        except TypeError:
            pass
    return config


def _get_dflash_config(config: Any) -> dict:
    if isinstance(config, dict):
        cfg = config.get("dflash_config", None)
    else:
        cfg = getattr(config, "dflash_config", None)
    if cfg is None:
        return {}
    if isinstance(cfg, dict):
        return cfg
    try:
        return dict(cfg)
    except Exception:
        return {}


def _parse_optional_int(
    value: Any,
    *,
    field_name: str,
    min_value: Optional[int] = None,
) -> Optional[int]:
    if value is None:
        return None
    try:
        parsed = int(value)
    except Exception as e:
        raise ValueError(f"Invalid {field_name}={value!r}.") from e
    if min_value is not None and parsed < int(min_value):
        comparator = "positive" if int(min_value) == 1 else f">= {int(min_value)}"
        raise ValueError(f"{field_name} must be {comparator}, got {parsed}.")
    return parsed


@dataclass(frozen=True)
class DFlashDraftConfig:
    num_hidden_layers: Optional[int]
    num_target_layers: Optional[int]
    block_size: Optional[int]
    target_layer_ids: Optional[List[int]]
    mask_token: str
    mask_token_id: Optional[int]

    def require_num_layers(self) -> int:
        if self.num_hidden_layers is None:
            raise ValueError(
                "DFLASH requires draft num_hidden_layers in config. "
                "Got config without num_hidden_layers."
            )
        return int(self.num_hidden_layers)

    def resolve_block_size(self, *, default: Optional[int] = None) -> Optional[int]:
        return self.block_size if self.block_size is not None else default

    def resolve_target_layer_ids(
        self,
        *,
        target_num_layers: int,
        draft_num_layers: Optional[int] = None,
    ) -> List[int]:
        target_num_layers = int(target_num_layers)
        if target_num_layers <= 0:
            raise ValueError(
                f"target_num_layers must be positive, got {target_num_layers}."
            )
        if self.target_layer_ids is None:
            if draft_num_layers is None:
                draft_num_layers = self.require_num_layers()
            return build_target_layer_ids(target_num_layers, int(draft_num_layers))

        resolved = list(self.target_layer_ids)
        if len(resolved) <= 0:
            raise ValueError(
                "DFLASH dflash_config.target_layer_ids must be non-empty. "
                f"Got len(target_layer_ids)={len(resolved)}."
            )
        for idx, val in enumerate(resolved):
            if val < 0 or val >= target_num_layers:
                raise ValueError(
                    "DFLASH target_layer_ids contains an out-of-range layer id. "
                    f"target_layer_ids[{idx}]={val}, target_num_layers={target_num_layers}."
                )
        return resolved


def parse_dflash_draft_config(*, draft_hf_config: Any) -> DFlashDraftConfig:
    dflash_cfg = _get_dflash_config(draft_hf_config)
    draft_text_config = _get_text_config(draft_hf_config)

    num_hidden_layers = _parse_optional_int(
        _cfg_get(draft_text_config, "num_hidden_layers", None),
        field_name="DFLASH draft num_hidden_layers",
        min_value=1,
    )
    raw_num_target_layers = dflash_cfg.get(
        "num_target_layers",
        _cfg_get(draft_hf_config, "num_target_layers", None),
    )
    num_target_layers = _parse_optional_int(
        raw_num_target_layers,
        field_name="DFLASH draft num_target_layers",
        min_value=1,
    )
    raw_block_size = dflash_cfg.get(
        "block_size",
        _cfg_get(draft_hf_config, "block_size", None),
    )
    block_size = _parse_optional_int(
        raw_block_size,
        field_name="DFLASH block_size",
        min_value=1,
    )

    layer_ids = dflash_cfg.get(
        "target_layer_ids",
        _cfg_get(draft_hf_config, "target_layer_ids", None),
    )
    parsed_target_layer_ids: Optional[List[int]]
    if layer_ids is None:
        parsed_target_layer_ids = None
    else:
        if not isinstance(layer_ids, (list, tuple)):
            raise ValueError(
                "DFLASH dflash_config.target_layer_ids must be a list of ints, "
                f"got type={type(layer_ids).__name__}."
            )
        parsed_target_layer_ids = [int(x) for x in layer_ids]
        if len(parsed_target_layer_ids) <= 0:
            raise ValueError(
                "DFLASH dflash_config.target_layer_ids must be non-empty. "
                f"Got len(target_layer_ids)={len(parsed_target_layer_ids)}."
            )

    mask_token = dflash_cfg.get("mask_token", None)
    if mask_token is None:
        mask_token = DEFAULT_DFLASH_MASK_TOKEN
    if not isinstance(mask_token, str) or not mask_token:
        raise ValueError(
            "DFLASH dflash_config.mask_token must be a non-empty string, "
            f"got {mask_token!r}."
        )

    mask_token_id = dflash_cfg.get("mask_token_id", None)
    if mask_token_id is not None:
        if not isinstance(mask_token_id, Integral) or isinstance(mask_token_id, bool):
            raise ValueError(
                "DFLASH dflash_config.mask_token_id must be an integer, "
                f"got {mask_token_id!r} (type={type(mask_token_id).__name__})."
            )
        mask_token_id = int(mask_token_id)
        if mask_token_id < 0:
            raise ValueError(
                "DFLASH dflash_config.mask_token_id must be non-negative, "
                f"got {mask_token_id}."
            )

    return DFlashDraftConfig(
        num_hidden_layers=num_hidden_layers,
        num_target_layers=num_target_layers,
        block_size=block_size,
        target_layer_ids=parsed_target_layer_ids,
        mask_token=mask_token,
        mask_token_id=mask_token_id,
    )


def compute_dflash_accept_len_and_bonus(
    *,
    candidates: torch.Tensor,
    target_predict: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if candidates.ndim != 2:
        raise ValueError(f"candidates must be 2D, got shape={tuple(candidates.shape)}")
    if target_predict.shape != candidates.shape:
        raise ValueError(
            "target_predict must have the same shape as candidates. "
            f"candidates.shape={tuple(candidates.shape)}, target_predict.shape={tuple(target_predict.shape)}"
        )

    bs, block_size = candidates.shape
    if bs <= 0:
        raise ValueError(f"batch size must be positive, got {bs}.")
    if block_size <= 0:
        raise ValueError(f"block_size must be positive, got {block_size}.")

    matches = candidates[:, 1:] == target_predict[:, :-1]
    accept_len = matches.to(torch.int32).cumprod(dim=1).sum(dim=1)
    bonus = target_predict[torch.arange(bs, device=target_predict.device), accept_len]
    return accept_len, bonus.to(torch.int64)
