from abc import ABC, abstractmethod
from ..core.quant_algo_info import QuantConfigManager


class BasePTQuantizer(ABC):
    def __init__(
        self,
        *args,
        slim_model,
        **kwargs,
    ):
        self.slim_model = slim_model
        self.quant_info = QuantConfigManager.get_config()
        self.ignore_layers = self.quant_info.ignore_layers
        self.quant_model_description = self.quant_info.quant_model_description

    # @property
    # def max_seq_length(self):
    #     global_config = GlobalConfig.get_config()
    #     calib_dataset_config = global_config.calib_dataset
    #     assert calib_dataset_config is not None, (
    #         "Configuration Error: Cannot access 'max_seq_length'. "
    #         "The 'calib_dataset' section is missing or None in the configuration, "
    #         "but this attribute requires it to be set."
    #     )
    #     assert (
    #         calib_dataset_config.dataset is not None
    #     ), "Configuration Error: 'calib_dataset.dataset' section is missing or None."
    #     return calib_dataset_config.dataset.max_seq_length

    # @property
    # def hidden_size(self):
    #     return self.slim_model.hidden_size

    # @property
    # def layers(self):
    #     return self.slim_model.layers

    @abstractmethod
    def get_qdq_module(self): ...

    @abstractmethod
    def prepare(self): ...

    @abstractmethod
    def calibrate(self, dataloader): ...

    @abstractmethod
    def convert(self): ...
