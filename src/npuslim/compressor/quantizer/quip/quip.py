
from npuslim.utils.factory import CompressorFactory
from ..gptq.gptq import GPTQ, GPTQConfig


__all__ = ["QUIP"]


class QuIPConfig(GPTQConfig):
    """QUIP algorithm specific configuration."""

    qfn_method: str = "rms" 
    preproc_rescale: bool = True
    preproc_proj: bool = True
    preproc_proj_mode: int = 2


@CompressorFactory.register("QuIP")
class QuIP(GPTQ):

    ConfigClass = QuIPConfig

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)