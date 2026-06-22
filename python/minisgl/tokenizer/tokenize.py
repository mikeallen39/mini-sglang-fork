from __future__ import annotations

import base64
import io
from pathlib import Path
from typing import Any, Dict, List
from urllib.parse import urlparse

import torch
from minisgl.message import TokenizeMsg
from PIL import Image
from transformers import PreTrainedTokenizerBase


def _load_image_from_url(url: str) -> Image.Image:
    if url.startswith("data:"):
        _, payload = url.split(",", 1)
        return Image.open(io.BytesIO(base64.b64decode(payload))).convert("RGB")

    parsed = urlparse(url)
    if parsed.scheme in ("", "file"):
        path = Path(parsed.path if parsed.scheme == "file" else url)
        return Image.open(path).convert("RGB")

    raise ValueError(f"Unsupported image_url scheme: {url}")


class TokenizeManager:
    def __init__(self, tokenizer: PreTrainedTokenizerBase, processor: Any | None = None) -> None:
        self.tokenizer = tokenizer
        self.processor = processor
        self._strip_glm_think = False

    def _tokenize_multimodal(
        self,
        msg: TokenizeMsg,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
        if self.processor is None:
            raise ValueError("Multimodal request received but no processor is loaded")
        assert isinstance(msg.text, list)

        images: List[Image.Image] = []
        processor_messages: List[Dict[str, Any]] = []
        for raw_msg in msg.text:
            content = raw_msg["content"]
            if isinstance(content, str):
                processor_messages.append(
                    {
                        "role": raw_msg["role"],
                        "content": [{"type": "text", "text": content}],
                    }
                )
                continue

            parts: List[Dict[str, Any]] = []
            for part in content:
                if part["type"] == "text":
                    parts.append({"type": "text", "text": part["text"]})
                elif part["type"] == "image_url":
                    image = _load_image_from_url(part["image_url"]["url"])
                    images.append(image)
                    parts.append({"type": "image", "image": image})
                else:
                    raise ValueError(f"Unsupported content part: {part}")
            processor_messages.append({"role": raw_msg["role"], "content": parts})

        prompt = self.processor.apply_chat_template(
            processor_messages,
            tokenize=False,
            add_generation_prompt=True,
            **(msg.chat_template_kwargs or {}),
        )
        assert isinstance(prompt, str)
        inputs = self.processor(
            text=[prompt],
            images=images or None,
            return_tensors="pt",
        )
        input_ids = inputs["input_ids"].view(-1).to(torch.int32).cpu()
        pixel_values = inputs.get("pixel_values")
        image_grid_thw = inputs.get("image_grid_thw")
        mm_token_type_ids = inputs.get("mm_token_type_ids")
        return (
            input_ids,
            pixel_values.contiguous().cpu() if pixel_values is not None else None,
            image_grid_thw.contiguous().cpu() if image_grid_thw is not None else None,
            mm_token_type_ids.view(-1).to(torch.int32).cpu() if mm_token_type_ids is not None else None,
        )

    def tokenize(
        self, msgs: List[TokenizeMsg]
    ) -> List[tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]]:
        results: List[
            tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]
        ] = []
        # TODO: batch tokenization
        for msg in msgs:
            if isinstance(msg.text, list) and msg.text and isinstance(msg.text[0].get("content"), list):
                results.append(self._tokenize_multimodal(msg))
                continue
            if isinstance(msg.text, list):
                prompt = self.tokenizer.apply_chat_template(
                    msg.text,
                    tokenize=False,
                    add_generation_prompt=True,
                    **(msg.chat_template_kwargs or {}),
                )
                assert isinstance(prompt, str)
            else:
                prompt = msg.text
            input_ids: torch.Tensor = (  # type: ignore
                self.tokenizer.encode(prompt, return_tensors="pt")
            )
            results.append((input_ids.view(-1).to(torch.int32), None, None, None))
        return results
