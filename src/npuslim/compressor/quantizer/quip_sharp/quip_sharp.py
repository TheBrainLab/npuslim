"""
QuIP# (QuIP-Sharp) Main Algorithm Implementation.

QuIP# achieves state-of-the-art quantization in extreme compression regimes (≤4 bits)
using:
1. Randomized Hadamard Transform for incoherence processing
2. E8 lattice codebooks for vector quantization
3. Optional fine-tuning

Reference: https://arxiv.org/abs/2402.04396
"""

import copy
import torch
import torch.nn as nn
from dataclasses import dataclass
from functools import partial
from tqdm import tqdm
from loguru import logger

from npuslim.utils.factory import CompressorFactory
from npuslim.utils.utils import find_parent_layer_and_sub_name, find_layers
from npuslim.utils.backend import bh
from npuslim.compressor.quantizer.base_algo import BaseCompressorAlgo
from npuslim.compressor.core.layer_wise_scheduler import (
    LayerWiseScheduler,
    execute_optimization_worker,
)

from .quip_sharp_config import QuIPSharpConfig
from .quip_sharp_module import QuIPSharpModule
from .quip_sharp_linear import QuIPSharpLinear
from .codebook import get_codebook

__all__ = ["QuIPSharp", "QuIPSharpConfig"]


@CompressorFactory.register("QuIPSharp")
class QuIPSharp(BaseCompressorAlgo):
    """
    QuIP# (QuIP-Sharp) implementation for extreme LLM compression.

    Key features:
    - 2-bit, 3-bit, 4-bit quantization using E8 lattice codebooks
    - Randomized Hadamard Transform for incoherence processing
    - Layer-wise optimization with LDLQ algorithm
    - Optional fine-tuning for improved fidelity

    Reference: https://arxiv.org/abs/2402.04396 (ICML 2024)
    """

    ConfigClass = QuIPSharpConfig

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.scheduler = None
        self.quantizer_results = {}
        self.codebook = None

    def prepare(self):
        """
        Preparation phase: Initialize codebook and layer-wise scheduler.
        """
        logger.info("🔧 [QuIP#] Preparing...")

        # Initialize codebook
        self.codebook = get_codebook(self.cfg.codebook, inference=False)
        logger.info(f"   Codebook: {self.cfg.codebook} (version {self.codebook.version})")
        logger.info(f"   Code size: {self.codebook.codesz}, Bits: {self.cfg.w_bits}")

        # Validate model dimensions
        self._validate_model_dimensions()

        # Set model to eval mode
        self.model.model.eval()

        if self.dataloader is None:
            raise ValueError("QuIP# algorithm requires a dataloader for calibration.")

        # Initialize scheduler for layer-wise processing
        self.scheduler = LayerWiseScheduler(self.model, self.dataloader)

        logger.info("✅ [QuIP#] Preparation completed.")

    def _validate_model_dimensions(self):
        """Check that linear layer dimensions are compatible with codebook."""
        all_layers = find_layers(self.model.model, [nn.Linear])
        incompatible = []

        for name, layer in all_layers.items():
            in_features = layer.in_features
            # Input dimension must be divisible by code size (8 for E8)
            if in_features % self.codebook.codesz != 0:
                incompatible.append(f"{name}: in_features={in_features}")

        if incompatible:
            logger.warning(
                f"⚠️ [QuIP#] Some layers have dimensions not divisible by {self.codebook.codesz}:\n"
                + "\n".join(f"   - {name}" for name in incompatible[:5])
            )
            if len(incompatible) > 5:
                logger.warning(f"   ... and {len(incompatible) - 5} more")

    def calibrate(self):
        """
        Calibration phase: Layer-wise quantization with LDLQ algorithm.
        """
        logger.info("⚖️ [QuIP#] Running layer-wise calibration...")

        # Define the optimization worker
        quant_worker = partial(
            execute_optimization_worker,
            method_name="fasterquant",
            results_dict=self.quantizer_results,
        )

        # Get layers for sequential processing
        layers = self.model.get_layers()

        self.scheduler.run(
            layers=layers,
            algo_class=QuIPSharpModule,
            process_fn=quant_worker,
            algo_config=self.cfg,
            ignore_layers=self.ignore_layers,
        )

        logger.success(
            f"✅ [QuIP#] Calibration completed. {len(self.quantizer_results)} layers optimized."
        )

    def convert(self):
        """
        Conversion phase: Replace linear layers with QuIPSharpLinear.

        For fake_quant mode, weights stay as float16 without packing.
        """
        if self.cfg.fake_quant:
            logger.info("🔄 [QuIP#] Fake quantization mode - skipping packing.")
            self.model.quantized = True
            return

        logger.info("🔄 [QuIP#] Converting modules and packing weights...")

        # Find all linear layers
        all_layers = find_layers(self.model.model, [nn.Linear])
        target_names = list(self.quantizer_results.keys())

        for name in tqdm(target_names, desc="Packing QuIP# Layers"):
            if name not in all_layers:
                continue

            ori_layer = all_layers[name]
            result = self.quantizer_results[name]

            # Get parent module for replacement
            parent_module, sub_name = find_parent_layer_and_sub_name(self.model.model, name)

            # Create QuIPSharpLinear layer
            new_layer = QuIPSharpLinear(
                in_features=ori_layer.in_features,
                out_features=ori_layer.out_features,
                codebook_name=self.cfg.codebook,
                codebook_version=self.codebook.version,
                bias=result.get("bias") is not None,
            )

            # Pack weights (must be on CPU for complex indexing)
            device = ori_layer.weight.device
            new_layer.cpu()
            ori_layer.cpu()

            new_layer.pack(
                Qidxs=result["Qidxs"],
                SU=result["SU"],
                SV=result["SV"],
                bias=result.get("bias"),
            )

            # Replace layer and move to device
            setattr(parent_module, sub_name, new_layer.to(device))

        # Cleanup
        bh.empty_cache()
        self.model.quantized = True
        self._update_model_config()

        logger.success("✅ [QuIP#] Packing completed.")

    def _update_model_config(self):
        """Update model config with QuIP# metadata."""
        self.model.model.config.quantization_config = {
            "quant_method": "quip_sharp",
            "bits": self.cfg.w_bits,
            "codebook": self.cfg.codebook,
            "codebook_version": self.codebook.version,
            "codesz": self.codebook.codesz,
            "incoh_mode": self.cfg.incoh_mode,
            "rescale_WH": self.cfg.rescale_WH,
        }
        logger.info("✅ QuIP# metadata updated in model config.")
