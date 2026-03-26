from ..base_model import BaseLLMModel
from npuslim.registry import ModelRegistry


@ModelRegistry.register("OPT", aliases=["OPTModel"])
class OPTSlimModel(BaseLLMModel):
    """OPT model support for quantization."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # OPT has both embed_tokens and embed_positions
        self.pre_transformer_module_names = [
            "model.decoder.embed_tokens",
            "model.decoder.embed_positions",
        ]
        self.block_name = "model.decoder.layers"
        self.post_transformer_module_names = [
            "model.decoder.final_layer_norm",
            "lm_head",
        ]
