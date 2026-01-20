import threadpoolctl as tctl
import os
from loguru import logger
from functools import partial
from torch import nn
from tqdm import tqdm
from huggingface_hub import save_torch_state_dict

from ..base_ptquantizer import BasePTQuantizer
from .gptq_module import GPTQModule
from .gptq_linear import GPTQQuantLinear
from npuslim.utils.factory import CompressorFactory
from npuslim.utils.utils import find_layers
from npuslim.compressor.helper.layer_wise_scheduler import LayerWiseScheduler


@CompressorFactory.register()
class INT4GPTQ(BasePTQuantizer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.scheduler = None

    def get_qdq_module(self): ...

    def prepare(self):
        self.quantizers = {}
        self.slim_model.model.eval()
        self.scheduler = LayerWiseScheduler(self.slim_model, self.dataloader)

    def calibrate(self):
        def quant_worker(layer_idx, handlers, subset, **kwargs):
            total_sub_layers = len(subset)
            for i, name in enumerate(subset.keys()):
                if name not in handlers:
                    logger.info(f"Layer {name} is skipped (not in handlers).")
                    continue
                logger.info(
                    f"-> [Layer {layer_idx}] Quantizing module ({i+1}/{total_sub_layers}): {name} | "
                )
                handler = handlers[name]
                scale, zero, g_idx = handler.quantize(
                    percdamp=kwargs.get("percdamp", 0.01),
                    blocksize=kwargs.get("blocksize", 128),
                    actorder=kwargs.get("actorder", True),
                )
                self.quantizers[name] = (scale, zero, g_idx)

        layers = self.slim_model.get_layers()
        info = self.quant_info
        quant_algo = partial(
            GPTQModule,
            quant_bits=info.w_quant_bits,
            group_size=info.group_size,
            sym=info.algo_specific_params.get("sym", True),
            static_groups=info.algo_specific_params.get("static_groups", True),
        )
        self.scheduler.run(
            layers=layers,
            algo_class=quant_algo,
            process_fn=quant_worker,
            ignore_layers=self.ignore_layers,
            # process_fn 的参数
            percdamp=info.algo_specific_params.get("percdamp", 0.01),
            blocksize=info.algo_specific_params.get("blocksize", 128),
            actorder=info.algo_specific_params.get("actorder", True),
        )
        info.target_quant_layers = list(self.quantizers.keys())

    def convert(self):
        logger.info("Packing GPTQ quantization parameters into model...")
        layers = find_layers(self.slim_model.model)
        layers = {n: layers[n] for n in self.quantizers}
        self._make_quant(
            self.slim_model.model,
            self.quantizers,
            self.quant_info.w_quant_bits,
            self.quant_info.group_size,
        )

        qlayers = find_layers(self.slim_model.model, [GPTQQuantLinear])

        with tctl.threadpool_limits(limits=1):
            pbar = tqdm(qlayers.keys(), leave=True)
            for name in pbar:
                pbar.set_description(f"Packing {name:<20.20s}")

                scale, zero, g_idx = self.quantizers[name]
                # so far can only pack layer on CPU
                layer_device = qlayers[name].device
                qlayers[name].cpu()
                layers[name], scale, zero, g_idx = (
                    layers[name].cpu(),
                    scale.cpu(),
                    zero.cpu(),
                    g_idx.cpu(),
                )
                qlayers[name].pack(layers[name], scale, zero, g_idx)
                qlayers[name].to(layer_device)

        logger.success("Packing GPTQ quantization parameters into model done.")

    def _make_quant(self, module, names, bits, group_size):
        if isinstance(module, GPTQQuantLinear):
            return

        for name, submodule in module.named_modules():
            if name in names:
                ori_layer_device = next(submodule.parameters()).device

                if isinstance(submodule, nn.Linear):
                    in_features = submodule.in_features
                    out_features = submodule.out_features
                bias = submodule.bias is not None
                new_layer = GPTQQuantLinear(
                    bits,
                    group_size,
                    in_features,
                    out_features,
                    bias,
                    weight_dtype=submodule.weight.dtype,
                )
                new_layer.device = ori_layer_device
                self._recurse_setattr(module, name, new_layer.to(ori_layer_device))

    def _recurse_setattr(self, module, name, value):
        """A function to recursively set attributes to a module."""
        if "." not in name:
            setattr(module, name, value)
        else:
            name, rest = name.split(".", 1)
            self._recurse_setattr(getattr(module, name), rest, value)

    def save_model(self, save_dir: str):
        """save quantized model and configs to local disk"""
        self.slim_model.model.cpu()

        # Save model
        class EmptyModule(nn.Module):
            def __init__(self):
                super(EmptyModule, self).__init__()

            def forward(self, x):
                return x

        # Save model and config files with empty state dict
        self.slim_model.model.config.quantization_config = {
            "bits": self.quant_info.w_quant_bits,
            "checkpoint_format": "gptq",
            "desc_act": True,
            "group_size": self.quant_info.group_size,
            "quant_method": "gptq",
            "static_groups": True,
            "sym": True,
            "true_sequential": True,
        }
        self.slim_model.model.config.save_pretrained(
            save_dir, state_dict=EmptyModule().state_dict()
        )

        # Remove empty state dict
        default_paths = [
            f"{save_dir}/model.safetensors",
            f"{save_dir}/pytorch_model.bin",
        ]
        for path in default_paths:
            if os.path.exists(path):
                os.remove(path)

        save_torch_state_dict(
            state_dict=self.slim_model.model.state_dict(),
            save_directory=save_dir,
            max_shard_size="5GB",
            safe_serialization=True,
            force_contiguous=True,
            shared_tensors_to_discard=self.slim_model.model._tied_weights_keys,
        )
        # self.slim_model.model.config.torch_dtype = "float16"
        self.slim_model.model.config.to_json_file(os.path.join(save_dir, "config.json"))
