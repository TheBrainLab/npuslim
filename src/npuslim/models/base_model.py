from __future__ import annotations

import importlib
import re
from abc import ABC
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from loguru import logger

from npuslim.streaming import SafeTensorIndex, ShardTensorReader


def get_hub_class(model_hub: str, class_name: str):
    """
    Dynamically fetch the specified class from the given hub (hf/ms).

    :param model_hub: 'hf' (Hugging Face) or 'ms' (ModelScope)
    :param class_name: The name of the class to import, e.g., 'AutoModelForCausalLM'
    :return: The class object
    """
    hub_pkg_map = {"hf": "transformers", "ms": "modelscope"}

    if model_hub not in hub_pkg_map:
        raise ValueError(
            f"❌ Unsupported hub: {model_hub}. Supported hubs are 'hf' and 'ms'."
        )
    pkg_name = hub_pkg_map[model_hub]

    try:
        module = importlib.import_module(pkg_name)
        cls = getattr(module, class_name)
        return cls

    except ImportError:
        raise ImportError(
            f"⚠️ '{pkg_name}' not installed. Please run: pip install {pkg_name}"
        )
    except AttributeError:
        raise AttributeError(
            f"⚠️ Class '{class_name}' not found in '{pkg_name}'. "
            f"Please check the class name spelling or package version."
        )


class BaseLLMModel(ABC):
    """Base model wrapper with full and streaming runtime support."""

    def __init__(
        self,
        *args,
        path: str,
        model_hub: str = "hf",
        runtime_mode: str = "full",
        chunk_size: int = 1,
        model_kwargs: Optional[Dict[str, Any]] = None,
        tokenizer_kwargs: Optional[Dict[str, Any]] = None,
        auto_memory_budget_bytes: Optional[int] = None,
        **kwargs,
    ):
        self.path = Path(path)
        self.model_hub = model_hub

        self.model_kwargs: Dict[str, Any] = model_kwargs or {}
        self.tokenizer_kwargs: Dict[str, Any] = tokenizer_kwargs or {}
        self.auto_memory_budget_bytes = auto_memory_budget_bytes

        self.skip_layer_names = ["lm_head"]
        self.pre_transformer_module_names = ["model.embed_tokens"]
        self.observer_layer_classes = [torch.nn.Linear]
        self.block_name = "model.layers"
        self._layers_path = "model.layers"

        self.model = None
        self.tokenizer = None
        self.config = None
        self.quantized = False
        self.model_type = "LLM"

        self.runtime_mode = "full"
        self.chunk_size = max(int(chunk_size), 1)

        self._safetensor_index: Dict[str, Any] = {}
        self._weight_map: Dict[str, str] = {}
        self._layer_tensor_map: Dict[int, List[str]] = {}
        self._chunk_cache: Dict[int, List[Dict[str, Any]]] = {}
        self._chunk_shards: Dict[int, set[str]] = {}
        self._tensor_reader: Optional[ShardTensorReader] = None
        self._total_layers: Optional[int] = None

        self.configure_runtime(runtime_mode, chunk_size=self.chunk_size)
        self.prepare(*args, **kwargs)

    def configure_runtime(self, mode: str, chunk_size: int = 1) -> None:
        mode = (mode or "full").lower()
        if mode not in {"full", "streaming", "auto"}:
            raise ValueError(f"Unsupported runtime mode: {mode}")
        self.runtime_mode = mode
        self.chunk_size = max(int(chunk_size), 1)

    def prepare(self, *args, **kwargs):
        _ = args, kwargs
        self.prepare_metadata()
        if self.runtime_mode == "full":
            self.prepare_full_model()

    def prepare_metadata(self) -> None:
        AutoTokenizer = get_hub_class(self.model_hub, "AutoTokenizer")

        logger.info(
            f"Loading tokenizer metadata from: '{self.path}' with kwargs: {self.tokenizer_kwargs}"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path=str(self.path),
            **self.tokenizer_kwargs,
        )

        try:
            AutoConfig = get_hub_class(self.model_hub, "AutoConfig")
            self.config = AutoConfig.from_pretrained(
                pretrained_model_name_or_path=str(self.path),
                **self.model_kwargs,
            )
        except Exception as exc:
            logger.warning(f"Failed to load AutoConfig metadata: {exc}")
            self.config = None

        self._load_safetensor_index()

        if self.runtime_mode == "auto":
            self.runtime_mode = self._resolve_auto_mode()
            logger.info(f"Resolved runtime mode to '{self.runtime_mode}'")

    def prepare_full_model(self) -> None:
        if self.model is not None:
            return

        AutoModelForCausalLM = get_hub_class(self.model_hub, "AutoModelForCausalLM")
        logger.info(
            f"Loading full model from: '{self.path}' with kwargs: {self.model_kwargs}"
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path=str(self.path),
            **self.model_kwargs,
        )

        if self.config is None:
            self.config = getattr(self.model, "config", None)

        if self.config is not None:
            logger.success(
                "Model, tokenizer, and config loaded successfully. "
                f"Model architecture: {self.config.architectures[0] if hasattr(self.config, 'architectures') and self.config.architectures else 'N/A'}"
            )

    def release_full_model(self) -> None:
        self.model = None
        self._chunk_cache.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _resolve_auto_mode(self) -> str:
        if self.auto_memory_budget_bytes is None:
            return "full"
        total_size = self._safetensor_index.get("metadata", {}).get("total_size")
        if total_size is None:
            return "full"
        return "streaming" if int(total_size) > int(self.auto_memory_budget_bytes) else "full"

    def _load_safetensor_index(self) -> None:
        index_path = self.path / "model.safetensors.index.json"
        if not index_path.exists():
            self._safetensor_index = {}
            self._weight_map = {}
            self._layer_tensor_map = {}
            self._infer_total_layers_from_config()
            return

        index = SafeTensorIndex(index_path)
        self._safetensor_index = {
            "metadata": {"total_size": index.total_size},
            "weight_map": index.weight_map,
        }
        self._weight_map = index.weight_map
        self._tensor_reader = ShardTensorReader(self.path)
        self._build_layer_tensor_map()

    def _infer_total_layers_from_config(self) -> None:
        if self.config is not None and hasattr(self.config, "num_hidden_layers"):
            self._total_layers = int(self.config.num_hidden_layers)

    def _build_layer_tensor_map(self) -> None:
        self._layer_tensor_map = {}
        max_idx = -1
        pattern = re.compile(rf"^{re.escape(self.block_name)}\.(\d+)\.")
        for tensor_name in self._weight_map:
            match = pattern.match(tensor_name)
            if not match:
                continue
            layer_idx = int(match.group(1))
            self._layer_tensor_map.setdefault(layer_idx, []).append(tensor_name)
            max_idx = max(max_idx, layer_idx)

        if max_idx >= 0:
            self._total_layers = max_idx + 1
        else:
            self._infer_total_layers_from_config()

    def get_total_layers(self) -> int:
        if self.model is not None:
            return len(self.get_layers())
        if self._total_layers is not None:
            return self._total_layers
        self._infer_total_layers_from_config()
        return self._total_layers or 0

    def get_chunk_count(self, chunk_size: Optional[int] = None) -> int:
        size = max(int(chunk_size or self.chunk_size), 1)
        total = self.get_total_layers()
        if total <= 0:
            return 0
        return (total + size - 1) // size

    def _load_tensor(self, tensor_name: str) -> torch.Tensor:
        shard = self._weight_map.get(tensor_name)
        if shard is None:
            raise KeyError(f"Tensor '{tensor_name}' not found in weight map")

        if self._tensor_reader is None:
            self._tensor_reader = ShardTensorReader(self.path)
        return self._tensor_reader.get_tensor(shard, tensor_name)

    def load_chunk(self, chunk_index: int, chunk_size: Optional[int] = None):
        size = max(int(chunk_size or self.chunk_size), 1)

        if self.runtime_mode == "full":
            self.prepare_full_model()
            layers = self.get_layers()
            start = chunk_index * size
            end = min(start + size, len(layers))
            return layers[start:end]

        start = chunk_index * size
        end = min(start + size, self.get_total_layers())
        payload: List[Dict[str, Any]] = []
        used_shards: set[str] = set()

        for layer_idx in range(start, end):
            layer_name = f"{self.block_name}.{layer_idx}"
            tensor_names = self._layer_tensor_map.get(layer_idx, [])
            for tensor_name in tensor_names:
                shard = self._weight_map.get(tensor_name)
                if shard:
                    used_shards.add(shard)
            tensors = {name: self._load_tensor(name) for name in tensor_names}
            payload.append({"name": layer_name, "index": layer_idx, "tensors": tensors})

        self._chunk_cache[chunk_index] = payload
        self._chunk_shards[chunk_index] = used_shards
        return payload

    def release_chunk(self, chunk_index: int) -> None:
        if chunk_index in self._chunk_cache:
            del self._chunk_cache[chunk_index]
        if self._tensor_reader is not None and chunk_index in self._chunk_shards:
            for shard_name in self._chunk_shards.pop(chunk_index):
                self._tensor_reader.release_shard(shard_name)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def forward(self, *args, **kwargs):
        if self.model is None:
            raise RuntimeError("Forward requires full model runtime. Call prepare_full_model().")
        return self.model(*args, **kwargs)

    def get_quant_convert_module(self):
        """
        Returns the module that will be converted to quantized.
        This is typically the main transformer module of the model.
        """
        return self.model

    def get_layers(self):
        if self.model is None:
            raise RuntimeError("Model is not loaded. Full mode is required for get_layers().")
        current = self.model
        for part in self._layers_path.split("."):
            current = getattr(current, part)
        return current

    def get_observer_layers(self, ignore_layers: list = []):
        """
        Default implementation: Returns all Linear layers except those in ignore_layers.
        Subclasses can override this for complex architectures (e.g., MoE).
        """
        if self.model is None:
            return {}

        all_modules = dict(self.model.named_modules())
        target_layers = {}
        for name, module in all_modules.items():
            if isinstance(module, torch.nn.Linear):
                if ignore_layers and any(ignored in name for ignored in ignore_layers):
                    continue
                target_layers[name] = module

        return target_layers

    def get_pre_transformer_modules(self):
        pre_transformer_modules_dict = {}
        if self.model is None:
            return pre_transformer_modules_dict

        for full_name in self.pre_transformer_module_names:
            current_module = self.model
            parts = full_name.split(".")
            for part in parts:
                if not hasattr(current_module, part):
                    current_module = None
                    break
                current_module = getattr(current_module, part)
            if current_module is not None:
                pre_transformer_modules_dict[full_name] = current_module
        return pre_transformer_modules_dict

    def get_kvcache_observer_layers_names(self, observe_names):
        names = ["self_attn.k_proj", "self_attn.v_proj"]
        return [
            k
            for k in observe_names
            if k.startswith(self.block_name)
            and k.split(".")[-2] + "." + k.split(".")[-1] in names
        ]
