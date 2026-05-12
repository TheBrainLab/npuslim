from npuslim.core import ModelRegistry


def test_model_registry_has_qwen3_vl_aliases():
    cls = ModelRegistry.get("Qwen3VL")
    assert cls.__name__ == "Qwen3VLSlimModel"

    alias_cls = ModelRegistry.get("Qwen3VLModel")
    assert alias_cls is cls
