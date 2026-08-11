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

    @property
    def moe_expert_fusion_map(self):
        """Describe how per-expert tensors fuse into 3D Parameters.

        Qwen3MoeExperts stores weights as 3D Parameters:
        - gate_up_proj [E, 2*intermediate, hidden]  (gate + up concatenated)
        - down_proj    [E, hidden, intermediate]

        This map tells the quantization pipeline to fuse per-expert 2D
        checkpoint tensors (experts.0.gate_proj + experts.0.up_proj) into
        the 3D format expected by the runtime model.
        """
        return {
            "gate_up_proj": (["gate_proj", "up_proj"], "cat"),
            "down_proj": (["down_proj"], "stack"),
        }
