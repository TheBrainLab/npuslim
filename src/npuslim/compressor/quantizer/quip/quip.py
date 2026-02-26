
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

from .quip_module import QuIPModule
from ..gptq.gptq_linear import GPTQQuantLinear  # Reuse GPTQ's linear layer

from npuslim.compressor.core.layer_wise_scheduler import (
    LayerWiseScheduler,
    execute_optimization_worker,
)

__all__ = ["QUIP", "QuIPConfig"]


@dataclass
class QuIPConfig:
    """QuIP (Quantization with Incoherence Processing) algorithm specific configuration."""

    # --- Basic Quantization Parameters ---
    w_bits: int = 4
    group_size: int = -1  # QuIP doesn't use grouping by default

    # --- QuIP-specific LDLQ Parameters ---
    # qfn: Quantization function - 'a' for minmax, 'b' for RMS (default with incoherence processing)
    qfn: str = "b"
    # qmethod: Quantization method - 'ldlq', 'ldl_gptqequiv'
    qmethod: str = "ldlq"
    # npasses: Number of greedy refinement passes (0 means no refinement)
    npasses: int = 0
    # unbiased: Use unbiased rounding
    unbiased: bool = False
    # blocksize: Block size for LDLQ processing
    blocksize: int = 128

    # --- Pre-processing Parameters (Incoherence Processing) ---
    preproc_rescale: bool = True  # Weight/Hessian diagonal rescaling
    preproc_proj: bool = True  # Random orthogonal projection
    preproc_proj_mode: int = 2  # Butterfly mode (2 = butterfly_nopermute)
    preproc_hessian: bool = True  # Use GPTQ-style Hessian computation

    # --- Fake Quantization (no packing) ---
    # Set to True for testing - outputs float16 weights without integer packing
    fake_quant: bool = True


@CompressorFactory.register("QuIP")
class QuIP(BaseCompressorAlgo):
    """
    QuIP (Quantization with Incoherence Processing) implementation.
    
    QuIP enhances GPTQ by adding incoherence processing:
    1. Weight/Hessian diagonal rescaling
    2. Random orthogonal projection (butterfly matrix)
    3. RMS quantization method (instead of minmax)
    
    Reference: https://github.com/Cornell-RelaxML/QuIP
    """
    ConfigClass = QuIPConfig

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.scheduler = None
        self.quantizer_results = {}

    def prepare(self):
        """
        Preparation phase: Set model to evaluation mode and initialize the layer-wise scheduler.
        """
        logger.info("🔧 [QuIP] Preparing LayerWise Scheduler...")
        self.model.model.eval()
        
        if self.dataloader is None:
            raise ValueError("QuIP algorithm requires a dataloader for calibration.")

        # Initialize scheduler to handle layer-by-layer activation capturing
        self.scheduler = LayerWiseScheduler(self.model, self.dataloader)

    def calibrate(self):
        """
        Calibration phase: Collect Hessian information and execute the QuIP optimization.
        """
        logger.info("⚖️ [QuIP] Running layer-wise calibration...")
        
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
            algo_class=QuIPModule,  # Use QuIPModule instead of GPTQModule
            process_fn=quant_worker,
            algo_config=self.cfg,
            ignore_layers=self.ignore_layers,
        )
        logger.success(f"✅ [QuIP] Calibration completed. {len(self.quantizer_results)} layers optimized.")

    def convert(self):
        """
        Conversion phase: Perform operator replacement and weight packing into low-bit formats.

        For fake quantization (fake_quant=True), this phase is skipped since weights
        are already quantized to float16 in the calibrate phase.
        """
        if self.cfg.fake_quant:
            logger.info("🔄 [QuIP] Fake quantization mode - skipping packing.")
            logger.info("   Weights are already quantized to float16 in calibrate phase.")
            self.model.quantized = True
            return

        logger.info("🔄 [QuIP] Converting modules and packing weights...")

        # 1. Identify layers that need to be replaced
        all_layers = find_layers(self.model.model, [nn.Linear])
        target_names = list(self.quantizer_results.keys())

        # 2. Iterate and execute replacement with progress tracking
        for name in tqdm(target_names, desc="Packing QuIP Layers"):
            if name not in all_layers:
                continue

            ori_layer = all_layers[name]
            scale, zero, g_idx = self.quantizer_results[name]

            # Retrieve parent module info for in-place replacement
            parent_module, sub_name = find_parent_layer_and_sub_name(self.model.model, name)

            # Initialize QuIP-specific QuantLinear layer (reuses GPTQQuantLinear)
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
        logger.success("✅ [QuIP] Packing completed.")

    def _update_model_config(self):
        """
        Synchronize quantization metadata with the model configuration for deployment compatibility.

        Note: We use "gptq" as quant_method for HuggingFace compatibility, since QuIP uses
        the same packing format as GPTQ.
        """
        self.model.model.config.quantization_config = {
            "bits": self.cfg.w_bits,
            "group_size": self.cfg.group_size,
            "sym": True,
            "quant_method": "gptq",
            "checkpoint_format": "gptq",
        }
        logger.info("✅ QuIP metadata updated in model config.")
