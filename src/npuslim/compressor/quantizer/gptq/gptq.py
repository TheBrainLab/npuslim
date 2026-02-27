import torch
import torch.nn as nn
import threadpoolctl as tctl
from loguru import logger
from functools import partial
from tqdm import tqdm
from dataclasses import dataclass

from npuslim.utils.factory import CompressorFactory
from npuslim.utils.utils import find_parent_layer_and_sub_name, find_layers
from npuslim.utils.backend import bh
from ..base_algo import BaseCompressorAlgo

from .gptq_module import GPTQModule
from .gptq_linear import GPTQQuantLinear

from npuslim.compressor.core.layer_wise_scheduler import (
    LayerWiseScheduler,
    execute_optimization_worker,
)

__all__ = ["GPTQ"]


@dataclass
class GPTQConfig:
    """GPTQ algorithm specific configuration."""
    # --- Basic Quantization Parameters ---
    w_bits: int = 4
    group_size: int = 128
    sym: bool = True

    # --- Pre-processing Parameters (Used during Observer find_params stage) ---
    preproc_rescale: bool = False
    preproc_proj: bool = False
    preproc_proj_mode: int = 0  # Mode selection for projection

    # --- Algorithm Execution Parameters (Used during GPTQModule optimization) ---
    blocksize: int = 128
    actorder: bool = True
    static_groups: bool = True
    percdamp: float = 0.01
    preproc_hessian: bool = True  # Whether to pre-process the Hessian matrix

    # --- Fake Quantization (no packing) ---
    # Set to True for testing - outputs float16 weights without integer packing
    fake_quant: bool = False


@CompressorFactory.register("GPTQ")
class GPTQ(BaseCompressorAlgo):
    """
    GPTQ (Accurate Post-Training Quantization) implementation.
    Utilizes Layer-wise Hessian information to minimize quantization error.
    """
    ConfigClass = GPTQConfig

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.scheduler = None
        self.quantizer_results = {}

    def prepare(self):
        """
        Preparation phase: Set model to evaluation mode and initialize the layer-wise scheduler.
        """
        logger.info("🔧 [GPTQ] Preparing LayerWise Scheduler...")
        self.model.model.eval()
        
        if self.dataloader is None:
            raise ValueError("GPTQ algorithm requires a dataloader for calibration.")

        # Initialize scheduler to handle layer-by-layer activation capturing
        self.scheduler = LayerWiseScheduler(self.model, self.dataloader)

    def calibrate(self):
        """
        Calibration phase: Collect Hessian information and execute the GPTQ optimization.
        """
        logger.info("⚖️ [GPTQ] Running layer-wise calibration...")
        
        # Define the optimization worker using partial to bind context
        quant_worker = partial(
            execute_optimization_worker,
            method_name="fasterquant",
            results_dict=self.quantizer_results,
        )

        # Retrieve layers (typically blocks/DecoderLayers) for sequential processing
        layers = self.model.get_layers()
        self.scheduler.run(
            layers=layers,
            algo_class=GPTQModule,
            process_fn=quant_worker,
            algo_config=self.cfg,
            ignore_layers=self.ignore_layers,
        )
        logger.success(f"✅ [GPTQ] Calibration completed. {len(self.quantizer_results)} layers optimized.")

    def convert(self):
        """
        Conversion phase: Perform operator replacement and weight packing into low-bit formats.
        """
        if self.cfg.fake_quant:
            logger.info("🔄 [GPTQ] Fake quantization mode - skipping packing.")
            logger.info("   Weights are already quantized to float16 in calibrate phase.")
            self.model.quantized = True
            return

        logger.info("🔄 [GPTQ] Converting modules and packing weights...")

        # 1. Identify layers that need to be replaced
        all_layers = find_layers(self.model.model, [nn.Linear])
        target_names = list(self.quantizer_results.keys())

        # 2. Iterate and execute replacement with progress tracking
        for name in tqdm(target_names, desc="Packing GPTQ Layers"):
            if name not in all_layers:
                continue
            
            ori_layer = all_layers[name]
            scale, zero, g_idx = self.quantizer_results[name]

            # Retrieve parent module info for in-place replacement
            parent_module, sub_name = find_parent_layer_and_sub_name(self.model.model, name)

            # Initialize GPTQ-specific QuantLinear layer
            new_layer = GPTQQuantLinear(
                bits=self.cfg.w_bits,
                group_size=self.cfg.group_size,
                infeatures=ori_layer.in_features,
                outfeatures=ori_layer.out_features,
                bias=ori_layer.bias is not None,
                weight_dtype=ori_layer.weight.dtype,
            )

            # Weight packing process (Must be executed on CPU to handle complex indexing)
            device = ori_layer.weight.device
            new_layer.cpu()
            ori_layer.cpu()
            new_layer.pack(ori_layer, scale.cpu(), zero.cpu(), g_idx.cpu())
            
            # Replace the original layer and move the new one back to the target device
            setattr(parent_module, sub_name, new_layer.to(device))

        # Cleanup memory and update model status
        bh.empty_cache()
        self.model.quantized = True
        self._update_model_config()
        logger.success("✅ [GPTQ] Packing completed.")

    def _update_model_config(self):
        """
        Synchronize quantization metadata with the model configuration for deployment compatibility.
        """
        self.model.model.config.quantization_config = {
            "bits": self.cfg.w_bits,
            "group_size": self.cfg.group_size,
            "sym": self.cfg.sym,
            "desc_act": self.cfg.actorder,
            "static_groups": self.cfg.static_groups,
            "quant_method": "gptq",
            "checkpoint_format": "gptq",
            "true_sequential": True,
        }
        logger.info("✅ GPTQ metadata updated in model config.")