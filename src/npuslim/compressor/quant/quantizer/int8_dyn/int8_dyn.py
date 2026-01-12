import json
import torch
from loguru import logger
from pathlib import Path
from safetensors.torch import load_file

from ..base_ptquantizer import BasePTQuantizer
from npuslim.utils.factory import CompressorFactory
from npuslim.utils.utils import find_parent_layer_and_sub_name
from npuslim.utils.backend import bh
from npuslim.compressor.quant.modules import INTDynQDQModule
from npuslim.compressor.quant.core.ptq_hook import PTQObserverHook
from npuslim.compressor.quant.observers import WEIGHT_OBSERVERS_CLASS


@CompressorFactory.register()
class INT8Dynamic(BasePTQuantizer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_qdq_module(self, sub_layer, name):
        weight_scale = None
        if name in self.weight_scales_dict:
            weight_scale = self.weight_scales_dict[name]

        q_linear = INTDynQDQModule(
            w_bits=8,
            weight=sub_layer.weight,
            weight_scale=weight_scale,
            bias=sub_layer.bias,
        )
        return q_linear

    def prepare(self):
        # add weight observer
        w_quant_method = self.quant_info.w_quant_method
        weight_observer = WEIGHT_OBSERVERS_CLASS.get(w_quant_method)
        self.quant_info.weight_observer = weight_observer

        # get observer layers names and kv names
        self.observer_layers = self.slim_model.get_observer_layers(self.ignore_layers)
        kv_names = self.slim_model.get_kvcache_observer_layers_names(
            self.observer_layers.keys()
        )
        self.quant_info.target_quant_layers = list(self.observer_layers.keys())
        self.quant_info.kv_names = kv_names

        # apply ptq hook
        self.ptq_hook = PTQObserverHook(
            model=self.slim_model, observer_layers=self.observer_layers
        )
        self.ptq_hook.apply_hook()
        self.weight_scales_dict = {}

    def calibrate(self): ...

    def convert(self):
        logger.info(
            ">>> [Quantization] Starting INT8 dynamic quantization conversion..."
        )
        for name, sub_layer in self.observer_layers.items():
            if (
                getattr(self.ptq_hook.observer_dict[sub_layer], "weight_observer")
                is not None
            ):
                if sub_layer.weight.device.type == "meta":
                    absolute_model_path = Path(
                        self.meta_cfg.meta_cfg.absolute_model_path
                    )
                    model_index_json = (
                        absolute_model_path / "model.safetensors.index.json"
                    )
                    with open(model_index_json, "r") as f:
                        model_index = json.load(f)

                    orign_w_file = (
                        absolute_model_path
                        / model_index["weight_map"][name + ".weight"]
                    )
                    orign_w = load_file(orign_w_file, device="cpu")
                    logger.info(f"Load meta weight {name} from file {orign_w_file}")
                    sub_layer.to_empty(device="cpu")
                    sub_layer.weight.data = orign_w[name + ".weight"]

                    if hasattr(sub_layer, "bias"):
                        if (name + ".bias") in model_index["weight_map"]:
                            orign_b_file = (
                                absolute_model_path
                                / model_index["weight_map"][name + ".bias"]
                            )
                            orign_b = load_file(orign_b_file, device="cpu")
                            logger.info(
                                f"Load meta bias {name} from file {orign_b_file}"
                            )
                            sub_layer.bias.data = orign_b[name + ".bias"]
                        else:
                            logger.info(
                                f"{name + '.bias'} not found. Set bias to None."
                            )
                            sub_layer.bias = None

                weight_scales = self.get_weight_scales(
                    sub_layer, self.ptq_hook.observer_dict[sub_layer].weight_observer
                )
                self.weight_scales_dict[name] = weight_scales

        self.ptq_hook.remove_hook()
        bh.empty_cache()
        self.ptq_hook.post_process()

        quant_convert_module = self.slim_model.get_quant_convert_module()
        for name, sub_layer in self.observer_layers.items():
            parent_layer, sub_name = find_parent_layer_and_sub_name(
                quant_convert_module, name
            )
            qdq_module = self.get_qdq_module(sub_layer, name)
            if qdq_module is not sub_layer:
                setattr(parent_layer, sub_name, qdq_module)

        self.slim_model.quantized = True
        logger.success(
            "<<< [Quantization] INT8 dynamic quantization conversion completed successfully."
        )
