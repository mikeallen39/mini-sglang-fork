import importlib

from .config import ModelConfig

_MODEL_REGISTRY = {
    "LlamaForCausalLM": (".llama", "LlamaForCausalLM"),
    "Qwen2ForCausalLM": (".qwen2", "Qwen2ForCausalLM"),
    "Qwen3ForCausalLM": (".qwen3", "Qwen3ForCausalLM"),
    "Qwen3MoeForCausalLM": (".qwen3_moe", "Qwen3MoeForCausalLM"),
    # GLM4 models
    "Glm4MoeLiteForCausalLM": (".glm4_moe_lite_hf", "Glm4MoeLiteForCausalLM"),
}


def get_model_class(model_architecture: str, model_config: ModelConfig):
    if model_architecture not in _MODEL_REGISTRY:
        raise ValueError(f"Model architecture {model_architecture} not supported")
    module_path, class_name = _MODEL_REGISTRY[model_architecture]
    if model_architecture == "Glm4MoeLiteForCausalLM" and model_config.use_mla_backend:
        # Use the dedicated MLA path only when MLA backend is explicitly enabled.
        module_path = ".glm4_moe_lite"
    module = importlib.import_module(module_path, package=__package__)
    model_cls = getattr(module, class_name)
    return model_cls(model_config)


__all__ = ["get_model_class"]
