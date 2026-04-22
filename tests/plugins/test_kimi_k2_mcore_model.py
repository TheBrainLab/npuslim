import importlib
import sys
import types
import unittest

import torch
from torch import nn


class _DummyLogger:
    def info(self, *args, **kwargs):
        return None

    def warning(self, *args, **kwargs):
        return None

    def debug(self, *args, **kwargs):
        return None

    def exception(self, *args, **kwargs):
        return None


class _FakeDeepseekV3ForCausalLM:
    def __init__(self, *args, **kwargs):
        return None

    def load_weights(self, weights):
        self._loaded_weights = list(weights)
        return {name for name, _ in self._loaded_weights}


class _FakeDeepseekV2Model(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()


class _FakeDeepseekV2DecoderLayer(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()


class KimiK2MCoreModelTests(unittest.TestCase):
    def setUp(self):
        self._saved_modules: dict[str, object] = {}
        self._module_name = "npuslim.plugins.vllm.model_executor.models.kimi_k2_mcore"

        self._patch_module("vllm", types.ModuleType("vllm"))

        vllm_logger = types.ModuleType("vllm.logger")
        vllm_logger.init_logger = lambda _name: _DummyLogger()
        self._patch_module("vllm.logger", vllm_logger)

        self._patch_module("vllm.model_executor", types.ModuleType("vllm.model_executor"))
        self._patch_module(
            "vllm.model_executor.models",
            types.ModuleType("vllm.model_executor.models"),
        )

        deepseek_v2 = types.ModuleType("vllm.model_executor.models.deepseek_v2")
        deepseek_v2.DeepseekV3ForCausalLM = _FakeDeepseekV3ForCausalLM
        deepseek_v2.DeepseekV2Model = _FakeDeepseekV2Model
        deepseek_v2.DeepseekV2DecoderLayer = _FakeDeepseekV2DecoderLayer
        deepseek_v2.DeepseekV2MLP = type("DeepseekV2MLP", (nn.Module,), {})
        deepseek_v2.DeepseekV2MoE = type("DeepseekV2MoE", (nn.Module,), {})
        deepseek_v2.support_torch_compile = lambda cls: cls
        deepseek_v2.RMSNorm = type("RMSNorm", (nn.Module,), {})
        deepseek_v2.Attention = type("Attention", (nn.Module,), {})
        deepseek_v2.QKVParallelLinear = type("QKVParallelLinear", (nn.Module,), {})
        deepseek_v2.RowParallelLinear = type("RowParallelLinear", (nn.Module,), {})
        deepseek_v2.current_platform = types.SimpleNamespace(device_type="cpu")
        self._patch_module("vllm.model_executor.models.deepseek_v2", deepseek_v2)

        sys.modules.pop(self._module_name, None)
        self.kimi_mod = importlib.import_module(self._module_name)

    def tearDown(self):
        sys.modules.pop(self._module_name, None)
        for name, old_mod in self._saved_modules.items():
            if old_mod is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = old_mod

    def _patch_module(self, name: str, module: object):
        if name not in self._saved_modules:
            self._saved_modules[name] = sys.modules.get(name)
        sys.modules[name] = module

    def test_model_inherits_native_deepseek_backend(self):
        model_cls = self.kimi_mod.KimiK2MCoreForCausalLM
        self.assertTrue(issubclass(model_cls, _FakeDeepseekV3ForCausalLM))

    def test_prepare_config_normalizes_gqa_fields(self):
        cfg = types.SimpleNamespace(
            num_query_groups=2,
            num_key_value_heads=128,
            q_lora_rank=1536,
            kv_lora_rank=512,
            qk_nope_head_dim=128,
            qk_rope_head_dim=64,
            v_head_dim=128,
            rope_theta=50000.0,
            rope_scaling={
                "type": "yarn",
                "factor": 32.0,
                "beta_fast": 1.0,
                "beta_slow": 1.0,
                "original_max_position_embeddings": 4096,
            },
        )

        self.kimi_mod._prepare_kimi_k2_mcore_hf_config(cfg)

        self.assertEqual(cfg.num_key_value_heads, 2)
        self.assertIsNone(cfg.q_lora_rank)
        self.assertEqual(cfg.kv_lora_rank, 0)
        self.assertEqual(cfg.qk_nope_head_dim, 0)
        self.assertEqual(cfg.qk_rope_head_dim, 0)
        self.assertEqual(cfg.v_head_dim, 0)
        self.assertEqual(cfg.rope_parameters["rope_theta"], 50000.0)
        self.assertEqual(cfg.rope_parameters["rope_type"], "yarn")
        self.assertEqual(cfg.rope_parameters["factor"], 32.0)

    def test_load_weights_zero_inits_optional_missing_bias(self):
        model_cls = self.kimi_mod.KimiK2MCoreForCausalLM
        model = model_cls.__new__(model_cls)

        q_ln_bias = torch.nn.Parameter(torch.ones(2))
        k_ln_bias = torch.nn.Parameter(torch.ones(2))
        gate_bias = torch.nn.Parameter(torch.ones(2))
        down_proj = torch.nn.Parameter(torch.full((2,), 3.0))

        named_params = [
            ("model.layers.0.self_attn.q_layernorm.bias", q_ln_bias),
            ("model.layers.0.self_attn.k_layernorm.bias", k_ln_bias),
            ("model.layers.0.mlp.gate.bias", gate_bias),
            ("model.layers.0.mlp.down_proj.weight", down_proj),
        ]
        model.named_parameters = lambda: iter(named_params)

        loaded = model.load_weights(
            [("model.layers.0.mlp.down_proj.weight", torch.ones(2))]
        )

        self.assertIn("model.layers.0.self_attn.q_layernorm.bias", loaded)
        self.assertIn("model.layers.0.self_attn.k_layernorm.bias", loaded)
        self.assertIn("model.layers.0.mlp.gate.bias", loaded)
        self.assertTrue(torch.equal(q_ln_bias.data, torch.zeros_like(q_ln_bias)))
        self.assertTrue(torch.equal(k_ln_bias.data, torch.zeros_like(k_ln_bias)))
        self.assertTrue(torch.equal(gate_bias.data, torch.zeros_like(gate_bias)))
        self.assertTrue(torch.equal(down_proj.data, torch.full((2,), 3.0)))


if __name__ == "__main__":
    unittest.main()
