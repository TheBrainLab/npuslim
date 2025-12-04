from .base_engine import BaseEngine
from ..utils.factory import QuantFactory

class PTQEngine(BaseEngine):
    def __init__(self):
        super().__init__()
        assert (
            "ptq" in self.cfg and self.cfg.ptq
        ), "Missing 'ptq' configuration in YAML."
        
        self.prepare_quant()

    def prepare_quant(self):
        quant_cfg = dict(self.cfg.ptq)
        self.quantizer = QuantFactory.create(**quant_cfg)
    
    def run(self):
        self.quantizer.calibrate(self.dataloader)