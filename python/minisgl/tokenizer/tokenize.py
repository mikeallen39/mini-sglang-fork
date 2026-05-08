from __future__ import annotations

from typing import List

import torch
from minisgl.message import TokenizeMsg
from transformers import PreTrainedTokenizerBase


class TokenizeManager:
    def __init__(self, tokenizer: PreTrainedTokenizerBase) -> None:
        self.tokenizer = tokenizer
        self._strip_glm_think = (
            getattr(tokenizer, "name_or_path", "").endswith("GLM-4.7-Flash")
            or tokenizer.convert_tokens_to_ids("[gMASK]") not in (-1, None)
        )

    def tokenize(self, msgs: List[TokenizeMsg]) -> List[torch.Tensor]:
        results: List[torch.Tensor] = []
        # TODO: batch tokenization
        for msg in msgs:
            if isinstance(msg.text, list):
                prompt = self.tokenizer.apply_chat_template(
                    msg.text,
                    tokenize=False,
                    add_generation_prompt=True,
                    **(msg.chat_template_kwargs or {}),
                )
                assert isinstance(prompt, str)
                if self._strip_glm_think and prompt.endswith("<think>"):
                    prompt = prompt[: -len("<think>")]
            else:
                prompt = msg.text
            input_ids: torch.Tensor = (  # type: ignore
                self.tokenizer.encode(prompt, return_tensors="pt")
            )
            results.append(input_ids.view(-1).to(torch.int32))
        return results
