from ..base_ptquantizer import BasePTQuantizer
from npuslim.utils.factory import CompressorFactory


@CompressorFactory.register()
class GPTQ(BasePTQuantizer):
    def __init__(
        self,
        bits: int = 4,
        quant_method: dict = dict(weight="per-group", group_size=128),
        ignore_layers: list = ["lm_head"],
    ):
        pass
