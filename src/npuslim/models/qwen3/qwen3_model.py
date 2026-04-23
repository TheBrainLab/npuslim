import re


from ..base_model import BaseLLMModel
from npuslim.core import ModelRegistry


@ModelRegistry.register("Qwen3", aliases=["Qwen3Model"])
class Qwen3SlimModel(BaseLLMModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pre_transformer_module_names = ["model.embed_tokens"]
        self.post_transformer_module_names = ["model.norm", "lm_head"]
        # for moe model
        self.skip_layer_names.append("model.layers.*.mlp.gate")
