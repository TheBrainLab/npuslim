from typing import TYPE_CHECKING
from abc import ABC, abstractmethod
from ..core.sparse_algo_info import SparseConfigManager
from npuslim.utils.config_parser import GlobalConfig

if TYPE_CHECKING:
    from npuslim.model.base_model import BaseLLMModel


class BaseSparser(ABC):
    def __init__(
        self,
        *args,
        slim_model: "BaseLLMModel",
        **kwargs,
    ):
        self.meta_cfg = GlobalConfig.get_config().meta
        self.slim_model = slim_model
        self.sparse_info = SparseConfigManager.get_config()
        self.ignore_layers = self.sparse_info.ignore_layers
        self.quant_model_description = self.sparse_info.quant_model_description

    def prepare(self): ...
    
    @abstractmethod
    def compress(self, dataloader): ...

    @abstractmethod
    def apply_masks(self): ...