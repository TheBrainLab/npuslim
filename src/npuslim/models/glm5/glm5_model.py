import torch

from ..base_model import BaseLLMModel
from npuslim.core import ModelRegistry


@ModelRegistry.register("GLM5", aliases=["Glm5Model", "GlmMoeDsa"])
class Glm5SlimModel(BaseLLMModel):
    """GLM-5 (GlmMoeDsa) model support for quantization.

    Architecture highlights:
      - MLA attention (q_a/q_b and kv_a/kv_b LoRA-style projections)
      - DSA indexer sub-module
      - Hybrid dense/MoE MLP: first_k_dense_replace layers use dense MLP,
        remaining layers use MoE with 256 routed + 1 shared experts.

    The original GlmMoeDsaExperts (3D Parameters: gate_up_proj [E,2I,H],
    down_proj [E,H,I]) is kept as-is. The streaming pipeline fuses expanded
    checkpoint tensors (experts.0.gate_proj.weight) into 3D before loading,
    quantizes per-expert via _ExpertSliceLinear, and saves fused 3D format
    (experts.gate_up_proj.weight) for vLLM compatibility.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pre_transformer_module_names = ["model.embed_tokens"]
        self.block_name = "model.layers"
        self.post_transformer_module_names = ["model.norm", "lm_head"]
        # MoE router gate should not be quantized
        self.skip_layer_names.append("model.layers.*.mlp.gate")

    @property
    def mtp_layer_count(self) -> int:
        """Number of MTP (Multi-Token Prediction) layers from config."""
        return getattr(self.config, "num_nextn_predict_layers", 0) or 0

    @property
    def mtp_layer_names(self) -> list[str]:
        """Names of MTP layers in the checkpoint (stored under model.layers.<N>)."""
        base = self.num_transformer_layers or 0
        return [f"{self.block_name}.{base + i}" for i in range(self.mtp_layer_count)]

    @property
    def mtp_extra_module_names(self) -> list[str]:
        """MTP-specific sub-modules that should not be quantized (norms, shared_head)."""
        names = []
        for mtp_name in self.mtp_layer_names:
            names.extend([
                f"{mtp_name}.enorm",
                f"{mtp_name}.hnorm",
                f"{mtp_name}.shared_head",
            ])
        return names

    @property
    def moe_expert_fusion_map(self):
        """Describe how per-expert tensors fuse into 3D Parameters.

        Returns a dict: fused_param_name -> (component_list, op)
        - op "cat":   concat components along dim 0, then stack experts along dim 0
        - op "stack": stack each component along dim 0 (expert dim)

        Consumed by BaseHessianAlgorithm._fuse_expert_tensors (pre-loading)
        and GPTQAlgorithm._refuse_moe_expert_tensors (post-quantization).
        """
        return {
            "gate_up_proj": (["gate_proj", "up_proj"], "cat"),
            "down_proj": (["down_proj"], "stack"),
        }

    def prepare_empty_model(self):
        model = super().prepare_empty_model()
        if model is not None:
            # Set dtype to match checkpoint dtype (e.g. bfloat16).
            # Without this, meta tensors default to float32, and
            # set_module_tensor_to_device upcasts bf16 weights to
            # float32, doubling GPU memory usage and causing OOM.
            config_dtype = getattr(self.config, "dtype", None)
            if config_dtype:
                dtype_map = {
                    "bfloat16": torch.bfloat16,
                    "float16": torch.float16,
                    "float32": torch.float32,
                }
                torch_dtype = dtype_map.get(str(config_dtype))
                if torch_dtype is not None:
                    model.to(torch_dtype)
        return model
