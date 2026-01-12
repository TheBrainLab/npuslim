import json
from typing import TYPE_CHECKING
from pathlib import Path
from abc import ABC, abstractmethod
from npuslim.utils.backend import bh
from npuslim.utils.config_parser import GlobalConfig
from ..core.quant_algo_info import QuantConfigManager

if TYPE_CHECKING:
    from npuslim.model.base_model import BaseLLMModel
    from torch.utils.data import DataLoader


class BasePTQuantizer(ABC):
    def __init__(
        self,
        *args,
        slim_model: "BaseLLMModel",
        dataloader: "DataLoader" = None,
        **kwargs,
    ):
        self.meta_cfg = GlobalConfig.get_config().meta
        self.quant_info = QuantConfigManager.get_config()

        self.slim_model = slim_model
        self.dataloader = dataloader
        self.ignore_layers = self.quant_info.ignore_layers
        self.quant_model_description = self.quant_info.quant_model_description

    def get_weight_scales(self, layer, weight_observer):
        weight = layer.weight.clone().detach()
        weight_observer(weight)
        return weight_observer.scales()

    @abstractmethod
    def get_qdq_module(self): ...

    @abstractmethod
    def prepare(self): ...

    @abstractmethod
    def calibrate(self): ...

    @abstractmethod
    def convert(self): ...

    def save_meta(self, save_path: Path | str):
        if bh.name == "npu":
            self.save_quant_model_description(save_path)

    def save_quant_model_description(self, save_path: Path | str):
        quant_algo = self.quant_info.quant_algo
        target_quant_layers = self.quant_info.target_quant_layers
        quant_model_description = dict(self.quant_info.quant_model_description)
        save_path = save_path / "quant_model_description.json"

        for key in self.slim_model.model.state_dict().keys():
            matched_layer = None
            for q_layer in target_quant_layers:
                if key.startswith(q_layer + "."):
                    matched_layer = q_layer
                    break

            if matched_layer:
                quant_model_description[key] = quant_algo
            else:
                quant_model_description[key] = "FLOAT"

        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(quant_model_description, f, indent=4, ensure_ascii=False)
