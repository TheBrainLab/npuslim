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
        quant_kwargs = dict(self.cfg.ptq)
        quant_kwargs.update(slim_model=self.slim_model)
        self.quantizer = QuantFactory.create(**quant_kwargs)
        self.quantizer.prepare()
    
    def run(self):
        self.quantizer.run(self.dataloader)