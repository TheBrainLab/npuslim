from types import SimpleNamespace

import torch

from npuslim.algorithms.quantization.gptq.gptq_algo import GPTQAlgorithm
from npuslim.core.backend import bh
from npuslim.registry import AlgorithmRegistry
from npuslim.tasks.compressor.context import ChunkContext, LayerInfo


def test_gptq_algorithm_registry():
    cls = AlgorithmRegistry.get("GPTQ")
    assert cls is GPTQAlgorithm


def test_gptq_update_metadata_cuda(monkeypatch):
    algo = GPTQAlgorithm(wbits=4, group_size=128, actorder=True)
    model_obj = SimpleNamespace(quantized=False)
    model_config = SimpleNamespace()
    algo.set_runtime_context(model_obj=model_obj, model_config=model_config)

    monkeypatch.setattr(bh, "name", "cuda")
    algo._update_quantization_metadata()

    assert model_obj.quantized is True
    assert hasattr(model_config, "quantization_config")
    qcfg = model_config.quantization_config
    assert qcfg["quant_method"] == "gptq"
    assert qcfg["checkpoint_format"] == "gptq"
    assert qcfg["bits"] == 4
    assert qcfg["group_size"] == 128


def test_gptq_update_metadata_npu(monkeypatch):
    algo = GPTQAlgorithm(wbits=4, group_size=128, actorder=True)
    model_obj = SimpleNamespace(quantized=False)
    model_config = SimpleNamespace(quantization_config={"old": True})
    algo.set_runtime_context(model_obj=model_obj, model_config=model_config)

    monkeypatch.setattr(bh, "name", "npu")
    algo._update_quantization_metadata()

    assert model_obj.quantized is True
    assert hasattr(model_config, "ascend_quant_config")
    acfg = model_config.ascend_quant_config
    assert acfg["model_quant_type"] == "W4A16"
    assert acfg["group_size"] == 128
    assert acfg["include_g_idx"] is True
    assert acfg["has_offset"] is True
    assert not hasattr(model_config, "quantization_config")


def test_gptq_pack_quant_linear_tensors_cuda(monkeypatch):
    monkeypatch.setattr(bh, "name", "cuda")
    algo = GPTQAlgorithm(wbits=4, group_size=2)

    linear = torch.nn.Linear(8, 4, bias=False)
    scales = torch.ones(4, 4, dtype=torch.float32)
    zeros = torch.ones(4, 4, dtype=torch.float32) * 8
    g_idx = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3], dtype=torch.int32)

    packed, packed_quant_names = algo._pack_quant_linear_tensors(
        module_name="self_attn.q_proj",
        linear_module=linear,
        scales=scales,
        zeros=zeros,
        g_idx=g_idx,
    )

    assert set(packed.keys()) == {
        "self_attn.q_proj.qweight",
        "self_attn.q_proj.qzeros",
        "self_attn.q_proj.scales",
        "self_attn.q_proj.g_idx",
    }
    assert set(packed_quant_names) == {
        "self_attn.q_proj.qweight",
        "self_attn.q_proj.qzeros",
        "self_attn.q_proj.scales",
        "self_attn.q_proj.g_idx",
    }


def test_gptq_process_chunk_does_not_require_full_model(monkeypatch):
    monkeypatch.setattr(bh, "name", "cuda")
    algo = GPTQAlgorithm(wbits=4, group_size=2, max_calib_samples=8)

    class _DummyLayer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.self_attn = torch.nn.Module()
            self.self_attn.q_proj = torch.nn.Linear(32, 32, bias=False)

        def forward(self, hidden_states, **kwargs):
            _ = kwargs
            return (self.self_attn.q_proj(hidden_states),)

    class _DummyRuntime(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = torch.nn.Module()
            self.model.embed_tokens = torch.nn.Embedding(128, 32)
            self.model.layers = torch.nn.ModuleList([_DummyLayer()])

        def forward(self, input_ids=None, **kwargs):
            _ = kwargs
            hidden = self.model.embed_tokens(input_ids)
            for layer in self.model.layers:
                hidden = layer(hidden)[0]
            return (hidden,)

    runtime_model = _DummyRuntime()
    model_obj = SimpleNamespace(
        block_name="model.layers",
        prepare_empty_model=lambda: runtime_model,
        release_empty_model=lambda: None,
    )
    model_config = SimpleNamespace()
    algo.set_runtime_context(model_obj=model_obj, model_config=model_config)

    chunk = ChunkContext(
        chunk_index=0,
        pre_modules=[
            SimpleNamespace(
                name="model.embed_tokens",
                tensors={"weight": torch.randn(128, 32, dtype=torch.float32)},
            )
        ],
        layers=[
            LayerInfo(
                name="model.layers.0",
                index=0,
                tensors={
                    "self_attn.q_proj.weight": torch.randn(32, 32, dtype=torch.float32),
                },
            )
        ],
        calib_data=[{"input_ids": torch.tensor([[1, 2, 3, 4]])}],
    )

    algo.on_start()
    out_chunk = algo.process_chunk(chunk)
    algo.on_finish()

    keys = set(out_chunk.layers[0].tensors.keys())
    assert "self_attn.q_proj.qweight" in keys
    assert "self_attn.q_proj.qzeros" in keys
    assert "self_attn.q_proj.scales" in keys
    assert "self_attn.q_proj.g_idx" in keys


def test_gptq_sanitize_layer_kwargs_disables_cache():
    class _DummyCache:
        pass

    kwargs = {
        "attention_mask": None,
        "use_cache": True,
        "past_key_values": _DummyCache(),
    }
    out = GPTQAlgorithm._sanitize_layer_kwargs(kwargs)
    assert out["use_cache"] is False
    assert out["past_key_values"] is None


def test_gptq_process_chunk_skips_unresolvable_moe_target(monkeypatch):
    monkeypatch.setattr(bh, "name", "cuda")
    algo = GPTQAlgorithm(wbits=4, group_size=2, max_calib_samples=8)

    class _NonSubscriptableExperts(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.act_fn = torch.nn.SiLU()

    class _DummyLayer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.self_attn = torch.nn.Module()
            self.self_attn.q_proj = torch.nn.Linear(32, 32, bias=False)
            self.mlp = torch.nn.Module()
            self.mlp.experts = _NonSubscriptableExperts()

        def forward(self, hidden_states, **kwargs):
            _ = kwargs
            return (self.self_attn.q_proj(hidden_states),)

    class _DummyRuntime(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = torch.nn.Module()
            self.model.embed_tokens = torch.nn.Embedding(128, 32)
            self.model.layers = torch.nn.ModuleList([_DummyLayer()])

        def forward(self, input_ids=None, **kwargs):
            _ = kwargs
            hidden = self.model.embed_tokens(input_ids)
            for layer in self.model.layers:
                hidden = layer(hidden)[0]
            return (hidden,)

    runtime_model = _DummyRuntime()
    model_obj = SimpleNamespace(
        block_name="model.layers",
        prepare_empty_model=lambda: runtime_model,
        release_empty_model=lambda: None,
    )
    model_config = SimpleNamespace()
    algo.set_runtime_context(model_obj=model_obj, model_config=model_config)

    chunk = ChunkContext(
        chunk_index=0,
        pre_modules=[
            SimpleNamespace(
                name="model.embed_tokens",
                tensors={"weight": torch.randn(128, 32, dtype=torch.float32)},
            )
        ],
        layers=[
            LayerInfo(
                name="model.layers.0",
                index=0,
                tensors={
                    "mlp.experts.0.down_proj.weight": torch.randn(32, 32, dtype=torch.float32),
                    "self_attn.q_proj.weight": torch.randn(32, 32, dtype=torch.float32),
                },
            )
        ],
        calib_data=[{"input_ids": torch.tensor([[1, 2, 3, 4]])}],
    )

    algo.on_start()
    out_chunk = algo.process_chunk(chunk)
    algo.on_finish()

    keys = set(out_chunk.layers[0].tensors.keys())
    assert "self_attn.q_proj.qweight" in keys
    assert "self_attn.q_proj.qzeros" in keys
    assert "self_attn.q_proj.scales" in keys
    assert "self_attn.q_proj.g_idx" in keys
    assert "mlp.experts.0.down_proj.weight" in keys


def test_gptq_on_start_calls_model_runtime_adapter(monkeypatch):
    monkeypatch.setattr(bh, "name", "cuda")
    algo = GPTQAlgorithm(wbits=4, group_size=2, max_calib_samples=8)

    class _DummyLayer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.self_attn = torch.nn.Module()
            self.self_attn.q_proj = torch.nn.Linear(32, 32, bias=False)

        def forward(self, hidden_states, **kwargs):
            _ = kwargs
            return (self.self_attn.q_proj(hidden_states),)

    class _DummyRuntime(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = torch.nn.Module()
            self.model.layers = torch.nn.ModuleList([_DummyLayer()])

    runtime_model = _DummyRuntime()
    called = {"ok": False}

    def _adapt(model):
        called["ok"] = True
        return model

    model_obj = SimpleNamespace(
        block_name="model.layers",
        prepare_empty_model=lambda: runtime_model,
        release_empty_model=lambda: None,
        adapt_gptq_runtime_model=_adapt,
    )
    model_config = SimpleNamespace()
    algo.set_runtime_context(model_obj=model_obj, model_config=model_config)

    algo.on_start()
    algo.on_finish()

    assert called["ok"] is True
