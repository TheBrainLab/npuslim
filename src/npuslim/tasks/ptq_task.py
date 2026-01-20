from loguru import logger
from pathlib import Path
from npuslim.utils.factory import CompressorFactory
from npuslim.compressor.quant.core.quant_algo_info import QuantConfigManager

from .base_task import BaseTask


class PTQTask(BaseTask):
    def __init__(self, model, config, ignore_layers, dataloader=None):
        self.model = model
        self.cfg = config
        self.dataloader = dataloader
        logger.info("Initializing PTQ Task components...")
        QuantConfigManager.initialize(
            quant_algo=self.cfg.type,
            quant_config=self.cfg.quant_config,
            ignore_layers=ignore_layers,
        )
        self.compressor = CompressorFactory.create(
            config=self.cfg, slim_model=self.model, dataloader=self.dataloader
        )
        if hasattr(self.compressor, "prepare"):
            self.compressor.prepare()

    def execute(self):
        logger.info("Executing PTQ calibration and conversion...")
        if self.dataloader is not None:
            logger.info("Running Calibration...")
            self.compressor.calibrate()
        else:
            logger.warning("No dataloader provided, skipping calibration.")

        self.compressor.convert()
        logger.success("PTQ Task execution finished.")

    def save_model(self, save_path: Path | str):
        if hasattr(self.compressor, "save_model"):
            return self.compressor.save_model(save_path)
        return False

    def save_meta(self, save_path: Path | str):
        if hasattr(self.compressor, "save_meta"):
            return self.compressor.save_meta(save_path)
