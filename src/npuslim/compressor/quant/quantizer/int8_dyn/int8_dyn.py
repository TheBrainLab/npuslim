import json
import torch
from loguru import logger
from pathlib import Path
from safetensors.torch import load_file

from ..base_ptquantizer import BasePTQuantizer
from npuslim.utils.factory import CompressorFactory
from npuslim.compressor.quant.modules import INTDynQDQModule
from npuslim.compressor.quant.core.ptq_hook import PTQObserverHook
from npuslim.utils.utils import find_parent_layer_and_sub_name


@CompressorFactory.register()
class INT8Dynamic(BasePTQuantizer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_qdq_module(self, sub_layer, name):
        weight_scale = None
        if name in self.slim_model.weight_scales_dict:
            weight_scale = self.slim_model.weight_scales_dict[name]

        q_linear = INTDynQDQModule(
            w_bits=8,
            weight=sub_layer.weight,
            weight_scale=weight_scale,
            bias=sub_layer.bias,
        )
        return q_linear

    def prepare(self):
        observer_layers_names = self.slim_model.get_observer_layers()
        kv_names = self.slim_model.get_kvcache_observer_layers_names(
            observer_layers_names.keys()
        )

        self.quant_info.observer_layers_names = observer_layers_names
        self.quant_info.kv_names = kv_names

        self.ptq_hook = PTQObserverHook(
            quant_model=self.slim_model, quant_algo_info=self.quant_info
        )
        self.ptq_hook.apply_hook()

    def calibrate(self, dataloader): ...

    def convert(self):
        logger.info(
            ">>> [Quantization] Starting INT8 dynamic quantization conversion..."
        )
        for name, sub_layer in self.quant_info.observer_layers_names.items():
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

                weight_scales = self.slim_model.get_weight_scales(
                    sub_layer, self.ptq_hook.observer_dict[sub_layer].weight_observer
                )
                self.slim_model.weight_scales_dict[name] = weight_scales

        self.ptq_hook.remove_hooks()
        torch.cuda.empty_cache()
        self.ptq_hook.post_process()

        quant_convert_module = self.slim_model.get_quant_convert_module()
        for name, sub_layer in self.quant_info.observer_layers_names.items():
            parent_layer, sub_name = find_parent_layer_and_sub_name(
                quant_convert_module, name
            )
            qdq_module = self.get_qdq_module(sub_layer, name)
            if qdq_module is not sub_layer:
                setattr(parent_layer, sub_name, qdq_module)

        for key in self.slim_model.model.state_dict().keys():
            self.quant_info.quantized_layers_names.append(key)

        self.slim_model.quantized = True
        logger.success(
            "<<< [Quantization] INT8 dynamic quantization conversion completed successfully."
        )