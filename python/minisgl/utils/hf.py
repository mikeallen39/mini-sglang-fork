import functools
import json
import os
from typing import Any

from huggingface_hub import snapshot_download
from minisgl.utils.logger import init_logger
from tqdm.asyncio import tqdm
from transformers import (
    AutoConfig,
    AutoProcessor,
    AutoTokenizer,
    PretrainedConfig,
    PreTrainedTokenizerBase,
)

logger = init_logger(__name__)


class DisabledTqdm(tqdm):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, disable=True)


def _normalize_multimodal_config_dict(config_dict: dict[str, Any]) -> dict[str, Any]:
    text_config = config_dict.get("text_config")
    if not isinstance(text_config, dict):
        return config_dict

    normalized = dict(text_config)
    if "tie_word_embeddings" not in normalized and "tie_word_embeddings" in config_dict:
        normalized["tie_word_embeddings"] = config_dict["tie_word_embeddings"]

    architectures = normalized.get("architectures") or config_dict.get("architectures") or []
    arch_map = {
        "Qwen3_5ForConditionalGeneration": "Qwen3_5ForCausalLM",
        "Qwen3_5MoeForConditionalGeneration": "Qwen3_5MoeForCausalLM",
    }
    normalized["architectures"] = [arch_map.get(arch, arch) for arch in architectures]
    return normalized


def load_tokenizer(model_path: str) -> PreTrainedTokenizerBase:
    try:
        return AutoTokenizer.from_pretrained(model_path, fix_mistral_regex=True)
    except TypeError:
        # Older transformers versions may not support fix_mistral_regex.
        return AutoTokenizer.from_pretrained(model_path)


def load_processor(model_path: str) -> Any:
    return AutoProcessor.from_pretrained(model_path)


@functools.cache
def _load_hf_config(model_path: str) -> Any:
    try:
        return AutoConfig.from_pretrained(model_path)
    except Exception as exc:
        logger.warning(
            "AutoConfig.from_pretrained failed for %s; falling back to local config.json parsing: %s",
            model_path,
            exc,
        )
        # Fallback for model types unknown to the installed transformers version
        # (e.g. glm4_moe_lite on older releases).
        config_file = os.path.join(model_path, "config.json")
        if not os.path.isfile(config_file):
            raise
        with open(config_file, "r", encoding="utf-8") as f:
            config_dict = json.load(f)
        config_dict = _normalize_multimodal_config_dict(config_dict)
        model_type = config_dict.get("model_type", "")
        fallback_cls = type("LocalFallbackConfig", (PretrainedConfig,), {"model_type": model_type})
        return fallback_cls(**config_dict)


def cached_load_hf_config(model_path: str) -> PretrainedConfig:
    config = _load_hf_config(model_path)
    return type(config)(**config.to_dict())


@functools.cache
def model_supports_multimodal(model_path: str) -> bool:
    from minisgl.models.config import ModelConfig

    return ModelConfig.from_hf(cached_load_hf_config(model_path)).is_multimodal


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
