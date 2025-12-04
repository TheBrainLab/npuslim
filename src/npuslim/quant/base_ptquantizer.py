from abc import ABC, abstractmethod
import torch.nn as nn
from loguru import logger
from typing import List, Set


class BasePTQuantizer(ABC):
    def __init__(self, slim_model, ignore_layers: List[str] = None, *args, **kwargs):
        default_ignore = ["lm_head", "embed_tokens", "final_layer_norm"]
        self.slim_model = slim_model
        self.ignore_layers = set(default_ignore)

        if ignore_layers:
            self.ignore_layers.update(ignore_layers)

        self.rollback_layers: List[str] = []
        self.quant_linear_layers: List[str] = []

    def prepare(self):
        self.rollback_layers, self.quant_linear_layers = self.rollback_ignore_layers(
            self.slim_model.model, self.ignore_layers
        )

    @staticmethod
    def rollback_ignore_layers(model, ignore_layers):
        all_linear_layers: List[str] = []
        all_conv_layers: List[str] = []
        all_module_layers: Set[str] = set()

        for name, module in model.named_modules():
            all_module_layers.add(name)

            if isinstance(
                module, (nn.Linear, nn.modules.linear.NonDynamicallyQuantizableLinear)
            ):
                all_linear_layers.append(name)

            elif isinstance(module, nn.Conv2d):
                all_conv_layers.append(name)

        if not all_linear_layers:
            logger.error(
                "FATAL ERROR: No nn.Linear found in the model. Cannot proceed with quantization."
            )
            raise ValueError("Quantization target missing (nn.Linear).")

        valid_ignore_layers = set()
        invalid_ignores = set()
        for name in ignore_layers:
            if name in all_module_layers:
                valid_ignore_layers.add(name)
            else:
                invalid_ignores.add(name)
        if invalid_ignores:
            logger.warning(
                f"The following names in 'ignore_layers' were not found in the model architecture"
                f"and will be skipped: {sorted(list(invalid_ignores))}"
            )

        initial_rollback_layers = valid_ignore_layers.copy()
        initial_rollback_layers.update(all_conv_layers)

        last_linear_layer = all_linear_layers[-1]
        if last_linear_layer not in initial_rollback_layers:
            logger.info(
                f"Automatically adding the last linear layer ({last_linear_layer}) to ignore list."
            )
            initial_rollback_layers.add(last_linear_layer)

        rollback_layers = list(initial_rollback_layers)
        quant_linear_layers = [
            name for name in all_linear_layers if name not in rollback_layers
        ]

        logger.info(f"Total nn.Linear layers: {len(all_linear_layers)}")
        logger.info(
            f"The following layers will maintain floating-point weights (rollback_layers):\n\t"
            + "\n\t".join(sorted(rollback_layers))
        )
        logger.info(
            f"The following layers will be quantized (quant_linear_layers):\n\t"
            + "\n\t".join(sorted(quant_linear_layers))
        )
        return rollback_layers, quant_linear_layers
