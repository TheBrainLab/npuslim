from abc import ABC
from typing import Optional, Dict, Any, List, TYPE_CHECKING

from npuslim.utils.utils import create_or_update_dataclass


if TYPE_CHECKING:
    from npuslim.model.base_model import BaseLLMModel
    from torch.utils.data import DataLoader


class BaseCompressorAlgo(ABC):
    """
    Abstract Base Class for all compression algorithms.
    This serves as a unified replacement for the legacy BaseSparser and BasePTQuantizer.
    """

    # Dataclass used for configuration validation; should be overridden by subclasses.
    ConfigClass = None

    def __init__(
        self,
        model: "BaseLLMModel",
        config: Dict[str, Any],
        dataloader: Optional["DataLoader"] = None,
        ignore_layers: List[str] = None,
        **kwargs,
    ):
        """
        Unified initialization interface invoked by CompressorFactory.
        """
        self.model = model
        self.dataloader = dataloader
        self.raw_config = config

        # 1. Automatic configuration parsing (Dict -> Dataclass)
        if self.ConfigClass:
            try:
                self.cfg = create_or_update_dataclass(self.ConfigClass, config)
            except Exception as e:
                raise ValueError(
                    f"Algo config parsing failed for {self.__class__.__name__}: {e}"
                )
        else:
            self.cfg = config

        # 2. Store global layer names to be ignored during the compression process
        self.ignore_layers = ignore_layers

    def prepare(self):
        """
        Preparation phase (e.g., allocating Hessian buffers, registering hooks).
        """
        pass

    def compress(self):
        """
        Entry point for sparsification execution.
        """
        raise NotImplementedError

    def apply_masks(self):
        """
        Entry point for applying sparse masks (Specific to Sparsification algorithms).
        """
        raise NotImplementedError

    def calibrate(self):
        """
        Entry point for calibration and statistics gathering (Specific to PTQ algorithms).
        """
        raise NotImplementedError

    def convert(self):
        """
        Entry point for module conversion and final quantization (Specific to PTQ algorithms).
        """
        raise NotImplementedError

    def _update_ascend_metadata(self):
        """
        Store Ascend-specific metadata for quant_model_description.json generation.

        Subclasses should override this method to define their Ascend format.
        The metadata is stored in model.config.ascend_quant_config and read by AscendSaver.

        Default implementation: No Ascend metadata (model treated as FLOAT by AscendSaver).
        """
        pass
