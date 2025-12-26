from loguru import logger
from pathlib import Path
from npuslim.utils.factory import CompressorFactory

from .base_task import BaseTask


class SparseTask(BaseTask):
    def __init__(self, model, config, ignore_layers, dataloader=None):
        self.model = model
        self.cfg = config
        self.dataloader = dataloader

        logger.info("Initializing Sparse Task components...")

        self.sparser = CompressorFactory.create(
            config=self.cfg, 
            slim_model=self.model
        )
        if hasattr(self.sparser, "prepare"):
            self.sparser.prepare()

    def execute(self):
        logger.info(f"Executing Sparsification ({self.cfg.type}) and mask application...")
        if self.dataloader is not None:
            logger.info(f"Running sparsification algorithm: {self.cfg.type}...")
            # 这里的 compress 相当于 PTQ 的 calibrate，负责计算权重的重要性并生成 Mask
            self.sparser.compress(self.dataloader)
        else:
            logger.warning("No dataloader provided, skipping data-driven sparsification.")
        self.sparser.apply_masks()
        
        logger.success("Sparse Task execution finished.")

    def save(self, save_path: Path):
        self.sparser.save(save_path)