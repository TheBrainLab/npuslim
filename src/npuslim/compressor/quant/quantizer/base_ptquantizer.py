import json
from typing import TYPE_CHECKING
from abc import ABC, abstractmethod
from npuslim.utils.config_parser import GlobalConfig
from ..core.quant_algo_info import QuantConfigManager

if TYPE_CHECKING:
    from npuslim.model.base_model import BaseLLMModel


class BasePTQuantizer(ABC):
    def __init__(
        self,
        *args,
        slim_model: "BaseLLMModel",
        **kwargs,
    ):
        self.meta_cfg = GlobalConfig.get_config().meta
        self.quant_info = QuantConfigManager.get_config()

        self.slim_model = slim_model
        self.ignore_layers = self.quant_info.ignore_layers
        self.quant_model_description = self.quant_info.quant_model_description

    @abstractmethod
    def get_qdq_module(self): ...

    @abstractmethod
    def prepare(self): ...

    @abstractmethod
    def calibrate(self, dataloader): ...

    @abstractmethod
    def convert(self): ...

    def save(self, save_path):
        quant_algo = self.quant_info.quant_algo
        observer_layers_names = self.quant_info.observer_layers_names
        quantized_layers_names = self.quant_info.quantized_layers_names
        quant_model_description = self.quant_info.quant_model_description
        save_path = save_path / "quant_model_description.json"

        for key in quantized_layers_names:
            matched_layer = None
            for q_layer in observer_layers_names:
                if key.startswith(q_layer + "."):
                    matched_layer = q_layer
                    break

            if matched_layer:
                quant_model_description[key] = quant_algo
            else:
                quant_model_description[key] = "FLOAT"

        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(quant_model_description, f, indent=4, ensure_ascii=False)
