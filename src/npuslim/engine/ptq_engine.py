from pathlib import Path
from loguru import logger

from .base_engine import BaseEngine
from ..compressor.quant.core.save import Saver
from npuslim.utils.config_parser import GlobalConfig
from npuslim.utils.factory import CompressorFactory
from npuslim.compressor.quant.core.quant_algo_info import QuantConfigManager


class PTQEngine(BaseEngine):
    def __init__(self):
        super().__init__()

        assert self.cfg.ptq is not None, (
            "PTQEngine requires 'ptq' configuration setting. "
            "Please ensure the YAML configuration file contains the 'ptq' section."
        )
        self.prepare_quant()

    def get_ignore_layers(self):
        default_ignore = {"lm_head", "embed_tokens", "final_layer_norm"}
        user_ignore = self.cfg.ptq.ignore_layers or []
        final_ignore = default_ignore.union(set(user_ignore))
        layers_str = "\n    - ".join(sorted(list(final_ignore)))
        logger.info(
            f"The following layers will be ignored during quantization:\n    - {layers_str}"
        )
        return final_ignore

    def prepare_quant(self):
        ignore_layers = self.get_ignore_layers()
        QuantConfigManager.initialize(
            quant_algo=self.cfg.ptq.type,
            quant_config=self.cfg.ptq.quant_config,
            ignore_layers=ignore_layers,
        )
        self.quantizer = CompressorFactory.create(
            config=self.cfg.ptq, slim_model=self.slim_model
        )
        self.slim_model.init_ptq()

        # 不同的压缩算法可能要做不同的准备
        if hasattr(self.quantizer, "prepare"):
            self.quantizer.prepare()

    def run(self):
        if self.dataloader is not None:
            self.quantizer.calibrate(self.dataloader)
        self.quantizer.convert()

    def save(self):
        # 某些量化算法的保存可能需要单独处理
        if hasattr(self.quantizer, "save"):
            self.quantizer.save()
        else:
            save_path = GlobalConfig.get_config().meta.save_path
            saver = Saver(save_model=self.slim_model, save_path=save_path)
            saver.save()
