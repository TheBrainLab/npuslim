import torch
from tqdm import tqdm
from loguru import logger
from ..base_ptquantizer import BasePTQuantizer
from ...utils.data_utils import batch_to_device
from ...utils.factory import QuantFactory

@QuantFactory.register()
class WXAX(BasePTQuantizer):
    def __init__(self, *args, w_bits, a_bits, **kwargs):
        super().__init__(*args, **kwargs)
        self.w_bits = w_bits
        self.a_bits = a_bits

    def prepare(self):
        super().prepare()
        self.ensure_model_on_npu(self.slim_model.model)

    @staticmethod
    def ensure_model_on_npu(model):
        try:
            model_device = next(model.parameters()).device
        except StopIteration:
            raise ValueError("Model has no parameters to check device placement.")

        if model_device.type != "npu":
            error_message = (
                f"W8A8 QUANTIZATION ERROR: Model is on '{model_device.type}'. NPU is REQUIRED for calibration."
                f"Please set `device_map` to `auto` or `npu` in your YAML configuration."
            )
            raise ValueError(error_message)

        logger.info(f"Model successfully verified to be on {model_device.type}.")

    def run(self, dataloader):
        self.slim_model.model.eval()
        logger.info(f"Starting WXAX calibration.")
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Calibrating"):
                batch = batch_to_device(batch, torch.device("npu:0"))
                self.slim_model.model(**batch)
