from ..base_model import BaseLLMModel
from ...utils.factory import ModelFactory

@ModelFactory.register("Qwen3")
class Qwen3ModelAdapter(BaseLLMModel):
    pass
