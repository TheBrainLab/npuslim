from dataclasses import dataclass, field, fields
from loguru import logger
from typing import List, Dict, Any, Optional


@dataclass
class SparseAlgoInfo:
    sparse_algo: str = field(
        default=None,
        metadata={"help": "Sparsification algorithm (e.g., 'sparsegpt', 'gmp')."},
    )
    sparsity: float = field(
        default=0.0, 
        metadata={"help": "Target sparsity ratio (e.g., 0.5 for 50%)."}
    )
    
    # --- N:M Sparsity (e.g., 2:4) ---
    prunen: int = field(
        default=0, 
        metadata={"help": "N for N:M structured pruning (the number of zeros)."}
    )
    prunem: int = field(
        default=0, 
        metadata={"help": "M for N:M structured pruning (the total window size)."}
    )
    
    # --- SparseGPT / Hessian Specific ---
    percdamp: float = field(
        default=0.01,
        metadata={"help": "Percent of the average Hessian diagonal to use for dampening."}
    )
    blocksize: int = field(
        default=128,
        metadata={"help": "Blocksize to use for adaptive mask selection / Hessian calculation."}
    )
    
    # --- 模型 Layers / 模块信息 ---
    ignore_layers: List[str] = field(
        default_factory=list,
        metadata={"help": "List of module names to skip during sparsification."}
    )
    processed_model_keys: List[str] = field(
        default_factory=list,
        metadata={"help": "Modules that have been successfully sparsified."}
    )
    quant_model_description: Dict[str, str] = field(
        default_factory=dict,
        metadata={"help": "Metadata about the sparsified model status."}
    )

    # --- 扩展参数 ---
    algo_specific_params: Dict[str, Any] = field(
        default_factory=dict,
        metadata={"help": "Algorithm-specific parameters (e.g., act_order)."},
    )


class SparseConfigManager:
    _config_instance: Optional["SparseAlgoInfo"] = None

    @classmethod
    def set_config(cls, config: "SparseAlgoInfo") -> None:
        if cls._config_instance is not None:
            logger.warning("Overwriting existing SparseAlgoInfo instance.")
        cls._config_instance = config

    @classmethod
    def get_config(cls) -> "SparseAlgoInfo":
        if cls._config_instance is None:
            raise RuntimeError("SparseAlgoInfo has not been initialized.")
        return cls._config_instance

    @classmethod
    def initialize(
        cls,
        sparse_algo: str,
        sparse_config: Dict[str, Any],
        ignore_layers: List[str] = ["lm_head"],
        algo_params: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> "SparseAlgoInfo":
        
        sparsity = sparse_config.get("sparsity", 0.0)
        pattern = sparse_config.get("pattern", "2:4")
        prunen = int(pattern.split(":")[0])
        prunem = int(pattern.split(":")[1])
        
        sparse_type = "unstructured"
        if prunem > 0:
            sparse_type = f"{prunen}:{prunem} structured"
            # 如果设置了 N:M，强制计算对应的 sparsity 比例（如 2:4 为 0.5）
            if prunen > 0:
                sparsity = float(prunen) / prunem

        percdamp = sparse_config.get("percdamp", 0.01)
        blocksize = sparse_config.get("blocksize", 128)
        quant_model_description = dict(
            version="1.0.0",
            model_quant_type=sparse_algo,
            group_size=blocksize,
        )

        config_data: Dict[str, Any] = {
            "sparse_algo": sparse_algo,
            "sparsity": sparsity,
            "prunen": prunen,
            "prunem": prunem,
            "percdamp": percdamp,
            "blocksize": blocksize,
            "ignore_layers": ignore_layers,
            "quant_model_description": quant_model_description,
            "algo_specific_params": algo_params if algo_params is not None else {},
        }

        # 合并额外的参数并过滤无效字段
        config_data.update(kwargs)
        valid_fields = {f.name for f in fields(SparseAlgoInfo)}
        config_data = {k: v for k, v in config_data.items() if k in valid_fields}

        config_instance = SparseAlgoInfo(**config_data)
        cls.set_config(config_instance)

        logger.info(f"Initialized SparseAlgoInfo: {sparse_algo} (Type: {sparse_type})")
        return config_instance