import json
from pathlib import Path

from npuslim.models.base_model import BaseLLMModel


class DummyModel(BaseLLMModel):
    """Test adapter that uses default BaseLLMModel behavior."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.block_name = "model.layers"


def _build_fake_hub(monkeypatch):
    from npuslim.models import base_model as base_model_module

    class FakeConfig:
        def __init__(self, n_layers=3):
            self.num_hidden_layers = n_layers
            self.architectures = ["FakeLM"]

        def save_pretrained(self, _):
            return None

    class FakeInner:
        def __init__(self):
            self.layers = ["layer0", "layer1", "layer2"]

    class FakeModel:
        def __init__(self):
            self.config = FakeConfig()
            self.model = FakeInner()

        def named_modules(self):
            return []

    class FakeAutoModelForCausalLM:
        calls = 0

        @classmethod
        def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
            _ = pretrained_model_name_or_path, kwargs
            cls.calls += 1
            return FakeModel()

    class FakeAutoTokenizer:
        calls = 0

        @classmethod
        def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
            _ = pretrained_model_name_or_path, kwargs
            cls.calls += 1
            return object()

    class FakeAutoConfig:
        calls = 0

        @classmethod
        def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
            _ = pretrained_model_name_or_path, kwargs
            cls.calls += 1
            return FakeConfig()

    mapping = {
        "AutoModelForCausalLM": FakeAutoModelForCausalLM,
        "AutoTokenizer": FakeAutoTokenizer,
        "AutoConfig": FakeAutoConfig,
    }
    monkeypatch.setattr(base_model_module, "get_hub_class", lambda _hub, name: mapping[name])
    return FakeAutoModelForCausalLM, FakeAutoTokenizer, FakeAutoConfig


def test_full_mode_loads_full_model_and_layers(monkeypatch, tmp_path):
    auto_model_cls, tokenizer_cls, _ = _build_fake_hub(monkeypatch)

    model = DummyModel(path=str(tmp_path), model_hub="hf", runtime_mode="full")

    assert auto_model_cls.calls == 1
    assert tokenizer_cls.calls == 1
    assert model.get_layers() == ["layer0", "layer1", "layer2"]


def test_streaming_mode_does_not_load_full_model(monkeypatch, tmp_path):
    auto_model_cls, tokenizer_cls, config_cls = _build_fake_hub(monkeypatch)

    model = DummyModel(path=str(tmp_path), model_hub="hf", runtime_mode="streaming")

    assert model.model is None
    assert auto_model_cls.calls == 0
    assert tokenizer_cls.calls == 1
    assert config_cls.calls == 1


def test_streaming_mode_reads_total_layers_from_index(monkeypatch, tmp_path):
    _build_fake_hub(monkeypatch)

    index_path = Path(tmp_path) / "model.safetensors.index.json"
    index_path.write_text(
        json.dumps(
            {
                "metadata": {"total_size": 1},
                "weight_map": {
                    "model.layers.0.self_attn.q_proj.weight": "model-00001.safetensors",
                    "model.layers.1.self_attn.q_proj.weight": "model-00001.safetensors",
                    "lm_head.weight": "model-00002.safetensors",
                },
            }
        ),
        encoding="utf-8",
    )

    model = DummyModel(path=str(tmp_path), model_hub="hf", runtime_mode="streaming")

    assert model.get_total_layers() == 2
