from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
from pathlib import Path
from loguru import logger

from npuslim.tasks.base_task import BaseTask
from npuslim.utils.factory import TaskFactory, CompressorFactory

# Explicitly define exported task classes
__all__ = ["PTQTask", "SparseTask"]


@dataclass
class CompressorTaskConfig:
    """
    Base configuration for compression-related tasks.
    """
    type: str           # Task category, e.g., 'ptq' or 'sparse'
    algo_name: str      # Specific algorithm identifier used by CompressorFactory
    ignore_layers: List[str] = field(default_factory=list) # Layer patterns to exclude
    algo_config: Dict[str, Any] = field(default_factory=dict) # Hyperparameters for the algorithm


@dataclass
class PTQTaskConfig(CompressorTaskConfig):
    """Configuration for Post-Training Quantization tasks."""
    pass


@dataclass
class SparseTaskConfig(CompressorTaskConfig):
    """Configuration for Model Sparsification tasks."""
    pass


class CompressorTask(BaseTask):
    """
    Intermediate base class for compression tasks. 
    Handles common algorithm initialization and layer exclusion logic.
    """
    def __init__(self, config: Dict[str, Any], resources: Dict[str, Any]):
        super().__init__(config, resources)
        self.algo = None
        # Resolve user-defined 'ignore_layers' patterns into actual model layer names
        raw_ignores = getattr(self.cfg, "ignore_layers", [])
        self.ignore_layers = self._resolve_layer_names(raw_ignores)

    def _init_algorithm(self):
        """
        Dynamically instantiates the core compression algorithm using CompressorFactory.
        The factory will perform a lazy import based on 'algo_name'.
        """
        logger.info(f"⚙️ Initializing algorithm: {self.cfg.algo_name}...")
        self.algo = CompressorFactory.create(
            algo_name=self.cfg.algo_name,
            model=self.model,
            config=self.cfg.algo_config,
            dataloader=self.dataloader,
            ignore_layers=self.ignore_layers,
        )
        # Standard hook for algorithm setup/pre-processing
        if hasattr(self.algo, "prepare"):
            self.algo.prepare()


@TaskFactory.register("ptq")
class PTQTask(CompressorTask):
    """
    Implements the standard PTQ workflow: Initialization -> Calibration -> Conversion.
    """
    ConfigClass = PTQTaskConfig

    def execute(self):
        # 1. Setup the algorithm (e.g., GPTQ, AWQ, INT8Dynamic)
        self._init_algorithm()

        logger.info("🚀 Executing PTQ Calibration and Conversion...")

        # 2. Calibration: Run statistics gathering if data is available
        if self.dataloader:
            logger.info("Running Calibration...")
            self.algo.calibrate()
        else:
            logger.warning("⚠️ No dataloader provided, skipping calibration phase.")

        # 3. Conversion: Finalize quantization (e.g., applying scales or weight packing)
        logger.info("Running Conversion...")
        self.algo.convert()

        logger.success("✨ [PTQTask] execution finished.")


@TaskFactory.register("sparse")
class SparseTask(CompressorTask):
    """
    Implements the sparsification workflow (e.g., Wanda, SparseGPT).
    """
    ConfigClass = SparseTaskConfig

    def execute(self):
        # 1. Setup the algorithm
        self._init_algorithm()

        logger.info(f"🚀 Executing Sparsification ({self.cfg.algo_name})...")
        
        # 2. Compression: Calculate masks/weights
        if self.dataloader:
            logger.info("Running data-driven sparsification...")
            self.algo.compress()
        else:
            logger.info("Running data-free sparsification...")
            if hasattr(self.algo, "compress"):
                self.algo.compress()

        # 3. Application: Prune the model weights based on computed masks
        if hasattr(self.algo, "apply_masks"):
            logger.info("Applying sparse masks to model weights...")
            self.algo.apply_masks()

        logger.success("✨ [SparseTask] execution finished.")