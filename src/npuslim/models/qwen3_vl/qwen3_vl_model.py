from npuslim.registry import ModelRegistry

from ..base_model import BaseLLMModel


@ModelRegistry.register(
    "Qwen3VL",
    aliases=[
        "Qwen3VLModel",
        "Qwen3VLMoe",
        "Qwen3VLMoeModel",
    ],
)
class Qwen3VLSlimModel(BaseLLMModel):
    """Qwen3-VL / Qwen3-VL-MoE model support."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_type = "VLM"
        self.block_name = "model.language_model.layers"
        self.pre_transformer_module_names = [
            "model.visual",
            "model.language_model.embed_tokens",
        ]
        self.post_transformer_module_names = [
            "model.language_model.norm",
            "lm_head",
        ]
        # Keep default INT8 scope on language blocks only.
        # Visual branch and embeddings/head are explicitly skipped.
        self.skip_layer_names.extend(
            [
                "model.visual.*",
                "model.language_model.embed_tokens",
            ]
        )
        # Router gate should not be quantized.
        self.skip_layer_names.append("model.language_model.layers.*.mlp.gate")

    def get_model_loader_candidates(self):
        model_type = getattr(self.config, "model_type", None)
        candidates = []
        if model_type == "qwen3_vl_moe":
            candidates.append("Qwen3VLMoeForConditionalGeneration")
        elif model_type == "qwen3_vl":
            candidates.append("Qwen3VLForConditionalGeneration")
        candidates.append("AutoModelForImageTextToText")
        return candidates

    def get_processor_loader_candidates(self):
        return ["AutoProcessor"]
