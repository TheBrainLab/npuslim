from dataclasses import dataclass
from functools import partial
from typing import Optional, Any

from loguru import logger

from npuslim.utils.factory import CompressorFactory
from npuslim.compressor.quantizer.base_algo import BaseCompressorAlgo
from npuslim.compressor.core.layer_wise_scheduler import (
    LayerWiseScheduler,
    execute_optimization_worker,
)

from npuslim.compressor.quantizer.sparsegpt.sparsegpt_module import SparseGPTModule


__all__ = ["SparseGPT"]


@dataclass
class SparseGPTConfig:
    """
    Configuration parameters for the SparseGPT algorithm.
    """

    # --- General Parameters ---
    blocksize: int = 128
    percdamp: float = 0.01

    # --- Mode A: Unstructured Sparsity ---
    # Only effective when prunem == 0
    sparsity: float = 0.5

    # --- Mode B: Semi-structured Sparsity (N:M) ---
    # e.g., 2:4 sparsity -> prunen=2, prunem=4
    # Default 0 disables N:M and falls back to unstructured sparsity
    prunen: int = 0
    prunem: int = 0

    # Hessian pre-processing settings
    preproc_hessian: bool = True
    preproc_rescale: bool = False
    preproc_proj: bool = False
    preproc_proj_mode: int = 0


@CompressorFactory.register("SparseGPT")
class SparseGPT(BaseCompressorAlgo):
    """
    SparseGPT pruning algorithm implementation.
    Inherits from BaseCompressorAlgo to leverage shared resources like 
    model, dataloader, and ignore_layers.
    """

    # Declare the configuration class for automatic parsing by the base class
    ConfigClass = SparseGPTConfig

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.scheduler: Optional[LayerWiseScheduler] = None

    def prepare(self):
        """
        Preparation Phase:
        1. Ensure the model is in evaluation mode.
        2. Initialize the LayerWiseScheduler to manage input capture and layer traversal.
        """
        logger.info("🛠️ [SparseGPT] Preparing model and scheduler...")

        # Must set to eval mode to disable Dropout, etc.
        self.model.model.eval()

        # LayerWiseScheduler handles hook management and input/output buffering
        self.scheduler = LayerWiseScheduler(self.model, self.dataloader)

    def compress(self):
        """
        Executes the core sparsification workflow.
        """
        logger.info(
            f"🚀 [SparseGPT] Starting compression (Sparsity={self.cfg.sparsity}, "
            f"prunen={self.cfg.prunen}, prunem={self.cfg.prunem})..."
        )

        # 1. Retrieve all optimizable layers (typically TransformerBlocks)
        layers = self.model.get_layers()

        # 2. Construct the Worker function
        # execute_optimization_worker is a generic layer-wise optimizer that
        # instantiates SparseGPTModule and calls the specified optimization method.
        sparse_worker = partial(
            execute_optimization_worker,
            method_name="fasterprune",
        )

        # 3. Launch the Scheduler
        # Key Points:
        # - ignore_layers: Uses the resolved list from the parent class (populated by Task).
        # - algo_config: Passes SparseGPTConfig directly to each SparseGPTModule instance.
        self.scheduler.run(
            layers=layers,
            algo_class=SparseGPTModule,  # Algorithm logic applied per layer
            process_fn=sparse_worker,    # The execution wrapper
            ignore_layers=self.ignore_layers,
            algo_config=self.cfg,
        )

        logger.success("✅ [SparseGPT] Compression completed.")

    def apply_masks(self):
        """
        Finalize mask application.
        Note: SparseGPT typically modifies weights in-place within 'fasterprune'.
        If the SparseGPTModule implementation stores masks in buffers rather 
        than modifying weights directly, they should be applied here.
        """
        logger.info("🔧 [SparseGPT] Masks are applied directly during compression.")
        pass