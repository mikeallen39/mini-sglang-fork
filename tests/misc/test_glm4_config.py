from __future__ import annotations

import torch
from transformers import PretrainedConfig

import minisgl.engine.engine as engine_module
import minisgl.engine.config as engine_config_module
from minisgl.distributed import DistributedInfo, build_ep_info, get_moe_tp_info
from minisgl.server.args import ServerArgs, parse_args


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
            moe_intermediate_size=2048,
            n_routed_experts=32,
            n_shared_experts=1,
            norm_topk_prob=True,
            num_attention_heads=20,
            num_experts_per_tok=4,
            num_hidden_layers=48,
            n_group=8,
            q_lora_rank=768,
            qk_nope_head_dim=192,
            qk_rope_head_dim=64,
            rms_norm_eps=1e-5,
            routed_scaling_factor=1.0,
            topk_group=2,
            v_head_dim=256,
            vocab_size=151552,
        )


class DummyDenseConfig(PretrainedConfig):
    model_type = "llama"

    def __init__(self):
        super().__init__(
            architectures=["LlamaForCausalLM"],
            hidden_act="silu",
            hidden_size=2048,
            intermediate_size=8192,
            max_position_embeddings=4096,
            num_attention_heads=16,
            num_hidden_layers=24,
            rms_norm_eps=1e-5,
            vocab_size=32000,
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
        ep_info=DistributedInfo(rank=0, size=1),
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
        ep_info=DistributedInfo(rank=0, size=1),
    )

    engine_module._adjust_config(config)

    assert config.attention_backend == "mla"
    assert config.moe_backend == "fused"


def test_parse_args_builds_ep_info(monkeypatch):
    monkeypatch.setattr("minisgl.utils.cached_load_hf_config", lambda _: DummyGlm4Config())

    config, _ = parse_args(
        ["--model-path", "dummy-glm4", "--tp-size", "2", "--ep-size", "2"]
    )

    assert config.tp_info == DistributedInfo(rank=0, size=2)
    assert config.ep_info == DistributedInfo(rank=0, size=2)


def test_build_ep_info_supports_divisor_ep_size():
    ep_infos = [build_ep_info(tp_rank=i, tp_size=4, ep_size=2) for i in range(4)]

    assert ep_infos == [
        DistributedInfo(rank=0, size=2),
        DistributedInfo(rank=0, size=2),
        DistributedInfo(rank=1, size=2),
        DistributedInfo(rank=1, size=2),
    ]

    moe_tp_infos = [
        get_moe_tp_info(tp_info=DistributedInfo(rank=i, size=4), ep_info=ep_infos[i])
        for i in range(4)
    ]
    assert moe_tp_infos == [
        DistributedInfo(rank=0, size=2),
        DistributedInfo(rank=1, size=2),
        DistributedInfo(rank=0, size=2),
        DistributedInfo(rank=1, size=2),
    ]


def test_adjust_config_rejects_ep_on_dense_model(monkeypatch):
    monkeypatch.setattr(
        engine_config_module,
        "cached_load_hf_config",
        lambda _: DummyDenseConfig(),
    )

    config = ServerArgs(
        model_path="dummy-dense",
        tp_info=DistributedInfo(rank=0, size=2),
        dtype=torch.bfloat16,
        ep_info=DistributedInfo(rank=0, size=2),
    )

    try:
        engine_module._adjust_config(config)
    except ValueError as exc:
        assert "MoE" in str(exc)
    else:
        raise AssertionError("Expected ep_size > 1 to be rejected for dense models")


def test_adjust_config_keeps_cuda_graph_for_fused_ep(monkeypatch):
    monkeypatch.setattr(
        engine_config_module,
        "cached_load_hf_config",
        lambda _: DummyGlm4Config(),
    )
    monkeypatch.setattr(engine_module.logger, "info_rank0", lambda *args, **kwargs: None)
    monkeypatch.setattr(engine_module.logger, "warning_rank0", lambda *args, **kwargs: None)

    config = ServerArgs(
        model_path="dummy-glm4",
        tp_info=DistributedInfo(rank=0, size=2),
        dtype=torch.bfloat16,
        ep_info=DistributedInfo(rank=0, size=2),
        cuda_graph_max_bs=32,
    )

    engine_module._adjust_config(config)

    assert config.moe_backend == "fused"
    assert config.cuda_graph_max_bs == 32


def test_adjust_config_accepts_divisor_ep_size(monkeypatch):
    monkeypatch.setattr(
        engine_config_module,
        "cached_load_hf_config",
        lambda _: DummyGlm4Config(),
    )
    monkeypatch.setattr(engine_module.logger, "info_rank0", lambda *args, **kwargs: None)
    monkeypatch.setattr(engine_module.logger, "warning_rank0", lambda *args, **kwargs: None)

    config = ServerArgs(
        model_path="dummy-glm4",
        tp_info=DistributedInfo(rank=0, size=4),
        dtype=torch.bfloat16,
        ep_info=DistributedInfo(rank=0, size=2),
        cuda_graph_max_bs=16,
    )

    engine_module._adjust_config(config)

    assert config.ep_info == DistributedInfo(rank=0, size=2)
    assert config.cuda_graph_max_bs == 16


def test_adjust_config_disables_cuda_graph_for_torch_ep(monkeypatch):
    monkeypatch.setattr(
        engine_config_module,
        "cached_load_hf_config",
        lambda _: DummyGlm4Config(),
    )
    monkeypatch.setattr(engine_module.logger, "info_rank0", lambda *args, **kwargs: None)
    monkeypatch.setattr(engine_module.logger, "warning_rank0", lambda *args, **kwargs: None)

    config = ServerArgs(
        model_path="dummy-glm4",
        tp_info=DistributedInfo(rank=0, size=2),
        dtype=torch.bfloat16,
        ep_info=DistributedInfo(rank=0, size=2),
        cuda_graph_max_bs=32,
        moe_backend="torch",
    )

    engine_module._adjust_config(config)

    assert config.cuda_graph_max_bs == 0
