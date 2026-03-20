import functools
import json
import os
from typing import Any

from huggingface_hub import snapshot_download
from tqdm.asyncio import tqdm
from transformers import AutoConfig, AutoTokenizer, PretrainedConfig, PreTrainedTokenizerBase


class DisabledTqdm(tqdm):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, disable=True)


def load_tokenizer(model_path: str) -> PreTrainedTokenizerBase:
    try:
        return AutoTokenizer.from_pretrained(model_path, fix_mistral_regex=True)
    except TypeError:
        # Older transformers versions may not support fix_mistral_regex.
        return AutoTokenizer.from_pretrained(model_path)


@functools.cache
def _load_hf_config(model_path: str) -> Any:
    try:
        return AutoConfig.from_pretrained(model_path)
    except Exception:
        # Fallback for model types unknown to the installed transformers version
        # (e.g. glm4_moe_lite on older releases).
        config_file = os.path.join(model_path, "config.json")
        if not os.path.isfile(config_file):
            raise
        with open(config_file, "r", encoding="utf-8") as f:
            config_dict = json.load(f)
        model_type = config_dict.get("model_type", "")
        fallback_cls = type("LocalFallbackConfig", (PretrainedConfig,), {"model_type": model_type})
        return fallback_cls(**config_dict)


def cached_load_hf_config(model_path: str) -> PretrainedConfig:
    config = _load_hf_config(model_path)
    return type(config)(**config.to_dict())


def download_hf_weight(model_path: str) -> str:
    if os.path.isdir(model_path):
        return model_path
    try:
        return snapshot_download(
            model_path,
            allow_patterns=["*.safetensors"],
            tqdm_class=DisabledTqdm,
        )
    except Exception as e:
        raise ValueError(
            f"Model path '{model_path}' is neither a local directory nor a valid model ID: {e}"
        )
