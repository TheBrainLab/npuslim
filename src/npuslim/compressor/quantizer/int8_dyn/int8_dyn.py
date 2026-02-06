import torch
import re
from loguru import logger
from tqdm import tqdm
from typing import Optional
from dataclasses import dataclass
from functools import partial

from npuslim.utils.factory import CompressorFactory
from npuslim.utils.utils import find_parent_layer_and_sub_name
from npuslim.utils.backend import bh
from ..base_algo import BaseCompressorAlgo
from ...observers import WEIGHT_OBSERVERS_CLASS
from ...utils.ptq_hook import PTQObserverHook

__all__ = ["INT8Dynamic"]


@dataclass
class INT8DynamicConfig:
    """
    Configuration for INT8 Dynamic Quantization.
    Generic parameters like 'ignore_layers' are handled by the base class.
    """

    w_bits: int = 8
    w_quant_method: str = "per-channel"  # Weights are typically quantized per-channel
    a_quant_method: str = (
        "per-token"  # Activation dynamic quantization is typically per-token
    )
    weight_observer: Optional[str] = (
        None  # If None, automatically inferred from w_quant_method
    )


@CompressorFactory.register("INT8Dynamic")
class INT8Dynamic(BaseCompressorAlgo):
    """
    INT8 Dynamic Quantization Algorithm.
    Weights are quantized statically, while activations are quantized dynamically during inference.
    """

    ConfigClass = INT8DynamicConfig

    def __init__(self, *args, **kwargs):
        # Base class handles config parsing, dataloader assignment, and ignore_layers resolution
        super().__init__(*args, **kwargs)

        self.weight_scales_dict = {}
        self.observer_layers = {}
        self.ptq_hook = None

    def prepare(self):
        """
        Preparation Phase: Identify target layers and initialize the appropriate Weight Observer.
        """
        logger.info("🔧 [INT8Dynamic] Preparing quantization environment...")

        # 1. Identify target modules (Linear layers)
        self.observer_layers = self.model.get_observer_layers(
            ignore_layers=self.ignore_layers
        )
        logger.info(f"   -> Found {len(self.observer_layers)} target layers.")

        # 2. Select Weight Observer Class (with fallback logic)
        # Priority: explicit 'weight_observer' > 'w_quant_method'
        obs_key = self.cfg.weight_observer or self.cfg.w_quant_method
        observer_cls = WEIGHT_OBSERVERS_CLASS.get(obs_key)

        if not observer_cls:
            available_keys = list(WEIGHT_OBSERVERS_CLASS.keys())
            raise ValueError(
                f"Weight observer key '{obs_key}' not found in registry.\n"
                f"Please check 'weight_observer' or 'w_quant_method' in your config.\n"
                f"Available keys: {available_keys}"
            )

        logger.info(
            f"   -> Using Weight Observer: {observer_cls.__name__} (Key: {obs_key})"
        )

        # 3. Mount PTQ Hooks
        # Use partial to pre-configure the observer factory
        w_obs_factory = partial(observer_cls, quant_bits=self.cfg.w_bits, group_size=-1)
        self.ptq_hook = PTQObserverHook(
            model=self.model,
            observer_layers=self.observer_layers,
            weight_observer=w_obs_factory,
        )
        self.ptq_hook.apply_hook()
        self.weight_scales_dict = {}

    def get_weight_scales(self, layer, weight_observer):
        """
        Core Helper: Uses the observer to calculate quantization scales for weights.
        """
        weight = layer.weight.clone().detach()
        weight_observer(weight)
        return weight_observer.scales()

    def calibrate(self, dataloader=None):
        """
        Calibration Phase: Dynamic quantization is data-free for weights;
        logic is integrated into convert(), so this remains empty.
        """
        pass

    def convert(self):
        """
        Conversion Phase: Replace standard Linear layers with Quantized-DeQuantized (QDQ) modules.
        """
        logger.info("🔄 [INT8Dynamic] Converting modules...")

        # 1. Lazy import of the specialized QDQ module to avoid heavy dependencies at startup
        from .int8_dyn_module import INTDynQDQModule

        # 2. Iterate through layers: Use the active Hook/Observer to get scales
        pbar = tqdm(self.observer_layers.items(), desc="Processing", unit="layer")
        for name, sub_layer in pbar:
            # Calculate Scale by manually driving the cached observer
            container = self.ptq_hook.observer_dict.get(sub_layer)
            weight_scale = self.get_weight_scales(sub_layer, container.weight_observer)

            # Locate parent module for replacement
            parent_layer, sub_name = find_parent_layer_and_sub_name(
                self.model.model, name
            )

            # Initialize the replacement module with quantized metadata
            qdq_module = INTDynQDQModule(
                w_bits=self.cfg.w_bits,
                weight=sub_layer.weight,
                weight_scale=weight_scale,
                bias=sub_layer.bias,
            )
            setattr(parent_layer, sub_name, qdq_module)

        # 3. Cleanup: Remove hooks and empty backend cache to free memory
        if self.ptq_hook:
            self.ptq_hook.remove_hook()
            self.ptq_hook.post_process()

        bh.empty_cache()

        # Update model status and metadata
        self.model.quantized = True
        self._update_model_config()
        logger.success("✅ [INT8Dynamic] Conversion completed.")

    def _update_model_config(self):
        """
        Metadata Synchronization: Strictly follows the 'compressed-tensors' standard
        for compatibility with inference engines like vLLM.
        """

        # 1. Parse strategy names from config strings
        w_strat = re.search(r"per-([a-zA-Z]+)", self.cfg.w_quant_method)
        w_strategy = w_strat.group(1) if w_strat else "channel"

        a_strat = re.search(r"per-([a-zA-Z]+)", self.cfg.a_quant_method)
        # Standardize strategy names (e.g., mapping 'per-token' to 'token')
        a_strategy = a_strat.group(1) if a_strat else "token"
        if a_strategy == "per-token":
            a_strategy = "token"

        # 2. Construct the nested configuration dictionary for deployment specs
        quantization_config = {
            "quant_method": "compressed-tensors",
            "quantization_status": "compressed",  # Required status flag
            "format": "int-quantized",
            "ignore": list(self.ignore_layers) if self.ignore_layers else [],
            "kv_cache_scheme": None,
            "config_groups": {
                "group_0": {
                    "targets": ["Linear"],
                    "weights": {
                        "num_bits": self.cfg.w_bits,
                        "strategy": w_strategy,
                        "dynamic": False,
                        "type": "int",
                    },
                    "input_activations": {
                        "num_bits": 8,
                        "strategy": a_strategy,
                        "dynamic": True,  # Marked as dynamic for activation
                        "type": "int",
                    },
                    "output_activations": None,
                }
            },
        }

        # 3. Inject metadata into the HuggingFace-style model config
        if not hasattr(self.model.model.config, "quantization_config"):
            self.model.model.config.quantization_config = {}

        self.model.model.config.quantization_config = quantization_config

        logger.info("✅ Quantization metadata updated.")
