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

    def get_parent_dict(self, observer_layers_dict):
        _ = observer_layers_dict
        return {}

    def get_layers(self):
        """Get transformer layers."""
        current = self.model
        for part in self.block_name.split("."):
            current = getattr(current, part)
        return current
