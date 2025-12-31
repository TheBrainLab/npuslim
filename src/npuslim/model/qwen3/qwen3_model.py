import re

from ..base_model import BaseLLMModel
from npuslim.utils.factory import ModelFactory
from npuslim.utils.utils import find_layers


@ModelFactory.register("Qwen3")
class Qwen3SlimModel(BaseLLMModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.block_name = "model.layers"

    def get_observer_layers(self, ignore_layers: list = []):
        names = [
            "k_proj",
            "v_proj",
            "q_proj",
            "o_proj",
            "up_proj",
            "gate_proj",
            "down_proj",
        ]
        observer_layers_dict = {}
        layers_dict = find_layers(self.model, layers=self.observer_layer_classes)
        for name, module in layers_dict.items():
            if name.startswith(self.block_name) and name.split(".")[-1] in names:
                if name not in ignore_layers:
                    observer_layers_dict[name] = module
        
        return observer_layers_dict
    
    def get_layers(self):
        return self.model.model.layers

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

    # def get_save_func(self): ...
