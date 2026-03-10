"""
Ascend-specific model saver for vLLM-Ascend deployment.

Generates quant_model_description.json alongside standard HuggingFace format.
"""

import json
from pathlib import Path
from typing import Dict, Any, Set
from loguru import logger

from .base_saver import BaseSaver
from .huggingface import HuggingFaceSaver
from .ascend_utils import (
    get_all_tensor_names,
    get_quantized_layer_names,
    build_tensor_quant_status,
)
from npuslim.utils.factory import SaverFactory


__all__ = ["AscendSaver"]


@SaverFactory.register("AscendSaver")
class AscendSaver(BaseSaver):
    """
    Ascend-optimized saver that generates quant_model_description.json.

    This saver:
    1. Delegates weight serialization to HuggingFaceSaver
    2. Generates quant_model_description.json for vLLM-Ascend
    3. Optionally strips quantization_config from model config

    The quantization algorithm populates `config.ascend_quant_config` with metadata.
    On NPU, GPTQQuantLinear stores weights in Ascend format directly (weight, weight_scale, weight_offset).
    """

    def _save_impl(self, save_path: Path):
        # 1. Save model weights via HuggingFaceSaver (already in Ascend format on NPU)
        hf_saver = HuggingFaceSaver(self.model, self.config)
        hf_saver._save_impl(save_path)

        # 2. Generate and save quant_model_description.json
        description = self._build_quant_description()
        desc_path = save_path / "quant_model_description.json"

        with open(desc_path, "w") as f:
            json.dump(description, f, indent=2)

        logger.info(f"   -> Generated quant_model_description.json")

        # 3. Optionally strip quantization_config for Ascend compatibility
        if self.config.get("strip_quant_config", True):
            self._strip_quantization_config(save_path)

    def _build_quant_description(self) -> Dict[str, Any]:
        """Build the quant_model_description.json structure."""
        ascend_config = getattr(self.model.model.config, "ascend_quant_config", None)

        if ascend_config is None:
            logger.info("   -> No Ascend metadata found, treating as FLOAT model")
            return self._build_float_description()

        return self._build_from_ascend_config(ascend_config)

    def _build_float_description(self) -> Dict[str, Any]:
        """Build description for non-quantized (float) models."""
        description = {
            "version": "1.0.0",
            "model_quant_type": "FLOAT",
        }

        all_tensors = get_all_tensor_names(self.model.model)
        for name in sorted(all_tensors):
            description[name] = "FLOAT"

        return description

    def _build_from_ascend_config(self, ascend_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Build description from algorithm-provided Ascend metadata.

        Args:
            ascend_config: Dict with keys:
                - model_quant_type: str (e.g., "W4A16", "W8A8_dynamic")
                - group_size: int (or -1 for per-channel)
                - quant_layer_types: List[str] of quantized layer class names
                - has_offset: bool
                - include_g_idx: bool
        """
        description = {
            "version": "1.0.0",
            "model_quant_type": ascend_config["model_quant_type"],
        }

        # Only include group_size if meaningful (>0)
        group_size = ascend_config.get("group_size", -1)
        if group_size > 0:
            description["group_size"] = group_size

        # Find quantized layers
        quant_layer_types = ascend_config.get("quant_layer_types", [])
        quant_layer_names = get_quantized_layer_names(
            self.model.model, quant_layer_types
        )

        logger.info(f"   -> Found {len(quant_layer_names)} quantized layers")

        # Build per-tensor status
        tensor_status = build_tensor_quant_status(
            self.model.model,
            quant_layer_names,
            quant_type=ascend_config["model_quant_type"],
            has_offset=ascend_config.get("has_offset", True),
            include_g_idx=ascend_config.get("include_g_idx", False),
        )
        description.update(tensor_status)

        # Log summary
        quant_count = sum(1 for v in tensor_status.values() if v != "FLOAT")
        logger.info(
            f"   -> Ascend config: {ascend_config['model_quant_type']}, "
            f"{quant_count} quantized tensors"
        )

        return description

    def _strip_quantization_config(self, save_path: Path):
        """Remove quantization_config from config.json for Ascend deployment."""
        config_path = save_path / "config.json"
        if not config_path.exists():
            return

        with open(config_path, "r") as f:
            config = json.load(f)

        if "quantization_config" in config:
            del config["quantization_config"]
            with open(config_path, "w") as f:
                json.dump(config, f, indent=2)
            logger.debug("   -> Stripped quantization_config from config.json")
