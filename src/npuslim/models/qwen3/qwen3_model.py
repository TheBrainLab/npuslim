import re


from ..base_model import BaseLLMModel
from npuslim.registry import ModelRegistry


@ModelRegistry.register("Qwen3", aliases=["Qwen3Model"])
class Qwen3SlimModel(BaseLLMModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.block_name = "model.layers"
        self._layers_path = "model.layers"
        # for moe model
        self.skip_layer_names.append("model.layers.*.mlp.gate")

    def get_parent_dict(self, observer_layers_dict):
        parent_mapping = {r"experts\.\d+": "experts"}
        parent_dict = {}
        for layer_name in observer_layers_dict.keys():
            parent_name = layer_name
            for k, v in parent_mapping.items():
                parent_name = re.sub(k, v, layer_name)
            if parent_name != layer_name:
                parent_dict[layer_name] = parent_name
        return parent_dict
