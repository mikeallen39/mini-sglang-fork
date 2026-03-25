from __future__ import annotations

import torch
from transformers import PretrainedConfig

import minisgl.engine.engine as engine_module
import minisgl.engine.config as engine_config_module
from minisgl.distributed import DistributedInfo
from minisgl.server.args import ServerArgs


class DummyGlm4Config(PretrainedConfig):
    model_type = "glm4_moe_lite"

    def __init__(self):
        super().__init__(
            architectures=["Glm4MoeLiteForCausalLM"],
            hidden_act="silu",
            hidden_size=2048,
            intermediate_size=8192,
            kv_lora_rank=512,
            max_position_embeddings=131072,
            num_attention_heads=20,
            num_experts_per_tok=4,
            num_hidden_layers=48,
            q_lora_rank=768,
            qk_nope_head_dim=192,
            qk_rope_head_dim=64,
            rms_norm_eps=1e-5,
            v_head_dim=256,
            vocab_size=151552,
        )


def test_server_args_preserve_glm4_mla_auto_detection(monkeypatch):
    monkeypatch.setattr(
        engine_config_module,
        "cached_load_hf_config",
        lambda _: DummyGlm4Config(),
    )

    config = ServerArgs(
        model_path="dummy-glm4",
        tp_info=DistributedInfo(rank=0, size=1),
        dtype=torch.bfloat16,
    )

    assert config.model_config.use_mla
    assert config.model_config.use_mla_backend


def test_glm4_auto_selects_mla_attention_and_fused_moe(monkeypatch):
    monkeypatch.setattr(
        engine_config_module,
        "cached_load_hf_config",
        lambda _: DummyGlm4Config(),
    )
    monkeypatch.setattr(engine_module.logger, "info_rank0", lambda *args, **kwargs: None)

    config = ServerArgs(
        model_path="dummy-glm4",
        tp_info=DistributedInfo(rank=0, size=1),
        dtype=torch.bfloat16,
    )

    engine_module._adjust_config(config)

    assert config.attention_backend == "mla"
    assert config.moe_backend == "fused"
