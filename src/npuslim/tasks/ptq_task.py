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
            config=self.cfg, 
            slim_model=self.model
        )
        self.model.init_ptq()
        if hasattr(self.compressor, "prepare"):
            self.compressor.prepare()

    def execute(self):
        logger.info("Executing PTQ calibration and conversion...")
        if self.dataloader is not None:
            logger.info("Running Calibration...")
            self.compressor.calibrate(self.dataloader)
        else:
            logger.warning("No dataloader provided, skipping calibration.")
            
        self.compressor.convert()
        logger.success("PTQ Task execution finished.")

    def save(self, save_path: Path):
        if self.compressor and hasattr(self.compressor, "save"):
            self.compressor.save(save_path)
