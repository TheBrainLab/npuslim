from dataclasses import dataclass, field, fields
from loguru import logger
from typing import List, Dict, Any, Type, Optional

from ..observers import (
    AbsMaxChannelWiseWeightObserver,
    # AbsMaxGroupWiseWeightObserver,
    AbsmaxPerchannelObserver,
    # AbsmaxPertensorObserver,
)


ACT_OBSERVERS_CLASS = {
    # "per-tensor": AbsmaxPertensorObserver,
    "per-channel": AbsmaxPerchannelObserver,
}
WEIGHT_OBSERVERS_CLASS = {
    # "per-tensor": AbsmaxPertensorObserver,
    "per-channel": AbsMaxChannelWiseWeightObserver,
    # "per-group": AbsMaxGroupWiseWeightObserver,
}


@dataclass
class QuantAlgoInfo:
    quant_algo: str = field(
        default=None,
        metadata={"help": "Quantization algorithm (e.g., 'int8_dynamic')."},
    )
    a_quant_bits: int = field(
        default=None, metadata={"help": "Bit-width for activation quantization."}
    )
    w_quant_bits: int = field(
        default=None, metadata={"help": "Bit-width for weight quantization."}
    )
    c_quant_bits: Optional[int] = field(
        default=None, metadata={"help": "Bit-width for kv Cache layer quantization."}
    )
    w_group_size: int = field(
        default=-1,
        metadata={
            "help": "Weight group size for group-wise quantization (-1 for per-channel)."
        },
    )
    a_quant_method: str = field(
        default=None,
        metadata={
            "help": "Method/granularity for activation quantization (e.g., 'per-channel')."
        },
    )
    w_quant_method: str = field(
        default=None,
        metadata={
            "help": "Method/granularity for weight quantization (e.g., 'per-channel')."
        },
    )
    c_quant_method: Optional[str] = field(
        default=None, metadata={"help": "Method/granularity for kv Cache quantization."}
    )

    # --- Observers ---
    act_observer: Optional[Type] = field(
        default=None,
        metadata={
            "help": "The observer class used for activation calibration (e.g., MinMaxObserver)."
        },
    )
    weight_observer: Optional[Type] = field(
        default=None,
        metadata={"help": "The observer class used for weight calibration."},
    )
    kv_cache_observer: Optional[Type] = field(
        default=None,
        metadata={
            "help": "The observer class used for KVCache calibration/optimization in attention layers."
        },
    )
    smooth_observer: Optional[Type] = field(
        default=None,
        metadata={
            "help": "The observer class used specifically for SmoothQuant/Aten smoothing calibration."
        },
    )

    # --- 模型 Layers / 模块信息 ---
    ignore_layers: List[str] = field(
        default_factory=list,
        metadata={
            "help": "List of full module names collected during initialization that should be explicitly skipped during quantization (but may still be observed)."
        },
    )
    kv_names: List[str] = field(
        default_factory=list,
        metadata={
            "help": "List of full names for KVCache projection layers (e.g., k_proj and v_proj) used for KVCache quantization/optimization."
        },
    )
    observer_layers_names: List[str] = field(
        default_factory=list,
        metadata={
            "help": "List of full names of all modules where a statistics observer has been registered for calibration."
        },
    )
    processed_model_keys: List[str] = field(
        default_factory=list,
        metadata={
            "help": "List of full names of modules that have been successfully converted to quantized format (e.g., weights transformed or operators replaced)."
        },
    )
    quant_model_description: Dict[str, str] = field(
        default_factory=dict,
        metadata={
            "help": "Detailed description of the quantized model, including version, "
            "global quant type, and per-layer quantization status."
        },
    )

    # --- 算法特定参数 ---
    algo_specific_params: Dict[str, Any] = field(
        default_factory=dict,
        metadata={
            "help": "Algorithm-specific parameters (e.g., 'zero_point', 'mse_range' for AWQ; 'scale_factors' for SmoothQuant)."
        },
    )


class QuantConfigManager:
    _config_instance: Optional["QuantAlgoInfo"] = None

    @classmethod
    def set_config(cls, config: "QuantAlgoInfo") -> None:
        if cls._config_instance is not None:
            logger.warning(
                "Overwriting existing QuantAlgoInfo instance. This should typically only happen once."
            )
        cls._config_instance = config

    @classmethod
    def get_config(cls) -> "QuantAlgoInfo":
        if cls._config_instance is None:
            raise RuntimeError(
                "QuantAlgoInfo has not been initialized. "
                "Please call QuantConfigManager.initialize() first."
            )
        return cls._config_instance

    @classmethod
    def initialize(
        cls,
        quant_algo: str,
        quant_config: Dict[str, Any],
        ignore_layers: List[str] = ["lm_head"],
        algo_params: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> "QuantAlgoInfo":
        quant_algo_lower = quant_algo.lower()
        is_dynamic = "dynamic" in quant_algo_lower

        weight_quant_method = quant_config.quant_method.get("weight", None)
        act_quant_method = quant_config.quant_method.get("activation", None)

        if not is_dynamic and act_quant_method is None:
            error_msg = (
                f"[{quant_algo}] is a static algorithm. "
                f"Please specify 'activation_method' in quant_config."
            )
            logger.error(error_msg)
            raise ValueError(error_msg)

        if weight_quant_method is None:
            error_msg = "[Error] 'weight' must be provided in quant_config."
            logger.error(error_msg)
            raise ValueError(error_msg)

        weight_observer = WEIGHT_OBSERVERS_CLASS.get(weight_quant_method)
        act_observer = ACT_OBSERVERS_CLASS.get(act_quant_method)
        # TODO: add kv cache observer and smooth observer
        kv_cache_observer = None
        smooth_observer = None

        a_bits = quant_config.get("a_bits", None)
        w_bits = quant_config.get("w_bits", None)
        c_bits = quant_config.get("c_bits", None)
        group_size = quant_config.get("group_size", -1)

        # add quant_model_description
        quant_model_description = dict(
            version="1.0.0",
            model_quant_type=quant_algo,
            group_size=group_size,
        )

        # TODO: add kv Cache quant config
        config_data: Dict[str, Any] = {
            "quant_algo": quant_algo,
            # 位宽
            "a_quant_bits": a_bits,
            "w_quant_bits": w_bits,
            "c_quant_bits": c_bits,
            # 粒度/方法
            "w_group_size": group_size,
            "a_quant_method": act_quant_method,
            "w_quant_method": weight_quant_method,
            "c_quant_method": None,
            # Observer 实例
            "act_observer": act_observer,
            "weight_observer": weight_observer,
            "kv_cache_observer": kv_cache_observer,
            "smooth_observer": smooth_observer,
            # layers 配置
            "ignore_layers": ignore_layers,
            "kv_names": kwargs.pop("kv_names", []),
            "observer_layers_names": kwargs.pop("observer_layers_names", []),
            "quant_model_description": quant_model_description,
            # 存储特定算法参数
            "algo_specific_params": algo_params if algo_params is not None else {},
        }

        config_data.update(kwargs)
        valid_fields = {f.name for f in fields(QuantAlgoInfo)}
        keys_to_remove = [key for key in config_data.keys() if key not in valid_fields]
        for key in keys_to_remove:
            logger.warning(
                f"Ignoring unknown parameter '{key}' passed to initialize_quant_info."
            )
            del config_data[key]

        config_instance = QuantAlgoInfo(**config_data)
        cls.set_config(config_instance)

        return config_instance
