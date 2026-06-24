from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class DFlashDraftInput:
    verified_id: torch.Tensor
    target_hidden: torch.Tensor
    ctx_lens: torch.Tensor
    draft_seq_lens: torch.Tensor


@dataclass
class DFlashVerifyInput:
    draft_token: torch.Tensor
    draft_token_num: int
    positions: torch.Tensor | None = None
    custom_mask: torch.Tensor | None = None
    topk: int = 1

    def __post_init__(self) -> None:
        if self.draft_token.ndim != 2:
            raise ValueError(
                "DFLASH verify draft_token must have shape [bs, block], "
                f"got {tuple(self.draft_token.shape)}."
            )
        if self.draft_token_num <= 0:
            raise ValueError(
                f"DFLASH verify draft_token_num must be positive, got {self.draft_token_num}."
            )
        if int(self.draft_token.shape[1]) != int(self.draft_token_num):
            raise ValueError(
                "DFLASH verify draft_token shape does not match draft_token_num: "
                f"shape={tuple(self.draft_token.shape)}, draft_token_num={self.draft_token_num}."
            )
