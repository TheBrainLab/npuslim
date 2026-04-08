from ..base_model import BaseLLMModel
from npuslim.registry import ModelRegistry


@ModelRegistry.register("GLM5", aliases=["Glm5Model", "GlmMoeDsa"])
class Glm5SlimModel(BaseLLMModel):
    """GLM-5 (GlmMoeDsa) model support for quantization.

    Architecture highlights:
      - MLA attention (q_a/q_b and kv_a/kv_b LoRA-style projections)
      - DSA indexer sub-module
      - Hybrid dense/MoE MLP: first_k_dense_replace layers use dense MLP,
        remaining layers use MoE with 256 routed + 1 shared experts.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pre_transformer_module_names = ["model.embed_tokens"]
        self.block_name = "model.layers"
        self.post_transformer_module_names = ["model.norm", "lm_head"]
        # MoE router gate should not be quantized
        self.skip_layer_names.append("model.layers.*.mlp.gate")
