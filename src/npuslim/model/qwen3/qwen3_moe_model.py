from .qwen3_model import Qwen3SlimModel
from npuslim.utils.factory import ModelFactory


@ModelFactory.register("Qwen3MoE")
class Qwen3MoESlimModel(Qwen3SlimModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.skip_layer_names.append("model.layers.*.mlp.gate")