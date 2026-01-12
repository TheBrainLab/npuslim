from loguru import logger
from pathlib import Path
from npuslim.utils.factory import CompressorFactory
from ..compressor.sparse.core.sparse_algo_info import SparseConfigManager

from .base_task import BaseTask


class SparseTask(BaseTask):
    def __init__(self, model, config, ignore_layers, dataloader=None):
        self.model = model
        self.cfg = config
        self.dataloader = dataloader

        logger.info("Initializing Sparse Task components...")
        SparseConfigManager.initialize(
            sparse_algo=self.cfg.type,
            sparse_config=self.cfg.sparse_config,
            ignore_layers=ignore_layers,
        )
        self.sparser = CompressorFactory.create(
            config=self.cfg, slim_model=self.model, dataloader=self.dataloader
        )
        if hasattr(self.sparser, "prepare"):
            self.sparser.prepare()

    def execute(self):
        logger.info(
            f"Executing Sparsification ({self.cfg.type}) and mask application..."
        )
        if self.dataloader is not None:
            logger.info(f"Running sparsification algorithm: {self.cfg.type}...")
            self.sparser.compress()
        else:
            logger.warning(
                "No dataloader provided, skipping data-driven sparsification."
            )
        self.sparser.apply_masks()

        logger.success("Sparse Task execution finished.")

    def save_model(self, save_path: Path | str):
        if hasattr(self.sparser, "save_model"):
            return self.sparser.save_model(save_path)
        return False

    def save_meta(self, save_path: Path | str):
        if hasattr(self.sparser, "save_meta"):
            return self.sparser.save_meta(save_path)
        return False
