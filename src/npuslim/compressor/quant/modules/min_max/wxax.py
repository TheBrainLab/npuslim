import torch
import gc

from torch import nn
from tqdm import tqdm
from loguru import logger

# from npuslim.compressor.quant.modules.quantizer import Quantizer, LinearQuantizer
from npuslim.compressor.quant.modules.base_ptquantizer import BasePTQuantizer
from npuslim.utils.factory import CompressorFactory


@CompressorFactory.register()
class WXAX(BasePTQuantizer):
    def __init__(
        self,
        *args,
        quant_config,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.quant_config = quant_config

    def prepare(self):
        super().prepare()
        self.prepare_model(self.slim_model.model)
        gc.collect()  # Manually trigger garbage collection.

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

    def prepare_model(self, model):
        self.ensure_model_on_npu(self.slim_model.model)

        def _set_module(ori_mod, submodule_key, module):
            tokens = submodule_key.split(".")
            sub_tokens = tokens[:-1]
            cur_mod = ori_mod
            for s in sub_tokens:
                cur_mod = getattr(cur_mod, s)
            setattr(cur_mod, tokens[-1], module)

        # Fake quant
        for name, module in model.named_modules():
            if name in self.rollback_layers:
                continue
            if isinstance(module, nn.Linear) or isinstance(
                module, nn.modules.linear.NonDynamicallyQuantizableLinear
            ):
                quant_mod = LinearQuantizer(quant_config=self.quant_config)
                quant_mod.set_param(module)
                _set_module(model, name, quant_mod)

        for name, module in model.named_modules():
            if name in self.rollback_layers:
                continue
            if isinstance(module, Quantizer):
                range_param = 0
                states_name = ".".join(name.split(".")[:-1])
                if states_name in self.act_states:
                    abs_max_tensor = max(
                        abs(self.act_states[states_name]["t_max"]),
                        abs(self.act_states[states_name]["t_min"]),
                    )
                    range_param = abs_max_tensor / self.act_states[states_name]["std"]
                module.enable_quantization(name, range_param)
                if module.is_input:
                    module.init_act_and_observer(act_method="min-max")

    @torch.no_grad()
    def run(self):
        self.slim_model.model.eval()
        logger.info(f"Starting WXAX calibration.")
        for batch in tqdm(self.dataloader):
            self.slim_model.model(**batch)
