from __future__ import annotations

import importlib
from abc import ABC
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from loguru import logger
from npuslim.core.backend import bh

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
        if model_hub == "ms":
            # ModelScope builds may not expose Auto* classes directly.
            try:
                module = importlib.import_module("transformers")
                return getattr(module, class_name)
            except Exception:
                pass
        raise AttributeError(
            f"⚠️ Class '{class_name}' not found in '{pkg_name}'. "
            f"Please check the class name spelling or package version."
        )


class BaseLLMModel(ABC):
    """Base model wrapper with metadata/full-model interfaces for task runtime."""

    def __init__(
        self,
        *args,
        path: str,
        model_hub: str = "hf",
        model_kwargs: Optional[Dict[str, Any]] = None,
        tokenizer_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        passthrough_model_keys = {
            "device_map",
            "torch_dtype",
            "revision",
            "low_cpu_mem_usage",
            "trust_remote_code",
            "attn_implementation",
            "max_memory",
        }
        passthrough_tokenizer_keys = {
            "revision",
            "use_fast",
            "trust_remote_code",
        }

        merged_model_kwargs: Dict[str, Any] = dict(model_kwargs or {})
        merged_tokenizer_kwargs: Dict[str, Any] = dict(tokenizer_kwargs or {})
        for key in list(kwargs.keys()):
            if key in passthrough_model_keys and key not in merged_model_kwargs:
                merged_model_kwargs[key] = kwargs.pop(key)
            if key in passthrough_tokenizer_keys and key not in merged_tokenizer_kwargs:
                merged_tokenizer_kwargs[key] = kwargs.pop(key)

        self.path = Path(path)
        self.path_str = path
        self.model_hub = model_hub
        self._remote_snapshot_dir: Optional[Path] = None

        self.model_kwargs: Dict[str, Any] = merged_model_kwargs
        self.tokenizer_kwargs: Dict[str, Any] = merged_tokenizer_kwargs

        self.skip_layer_names = ["lm_head"]
        self.block_name = "model.layers"
        self.pre_transformer_module_names = ["model.embed_tokens"]
        self.post_transformer_module_names = ["lm_head"]
        # self.observer_layer_classes = [torch.nn.Linear]

        self.model = None
        self.tokenizer = None
        self.config = None
        self.quantized = False
        self.model_type = "LLM"

        self.prepare_metadata()

    def _resolve_pretrained_source(self) -> str:
        if self.path.exists():
            return str(self.path)
        if self.model_hub == "ms":
            try:
                from modelscope.hub.snapshot_download import snapshot_download

                if self._remote_snapshot_dir is None or not self._remote_snapshot_dir.exists():
                    revision = self.tokenizer_kwargs.get("revision") or self.model_kwargs.get(
                        "revision"
                    )
                    snapshot_dir = snapshot_download(model_id=self.path_str, revision=revision)
                    self._remote_snapshot_dir = Path(snapshot_dir)
                return str(self._remote_snapshot_dir)
            except Exception as exc:
                logger.debug(
                    f"Failed to resolve ModelScope snapshot for '{self.path_str}': {exc}"
                )
        return self.path_str

    def prepare_metadata(self, pretrained_source: Optional[str] = None) -> None:
        source = pretrained_source or self._resolve_pretrained_source()
        AutoTokenizer = get_hub_class(self.model_hub, "AutoTokenizer")

        logger.info(
            f"Loading tokenizer metadata from: '{source}' with kwargs: {self.tokenizer_kwargs}"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path=source,
            **self.tokenizer_kwargs,
        )

        try:
            AutoConfig = get_hub_class(self.model_hub, "AutoConfig")
            self.config = AutoConfig.from_pretrained(
                pretrained_model_name_or_path=source,
                **self.model_kwargs,
            )
        except Exception as exc:
            logger.warning(f"Failed to load AutoConfig metadata: {exc}")
            self.config = None

    def prepare_full_model(self, pretrained_source: Optional[str] = None) -> None:
        if self.model is not None:
            return

        source = pretrained_source or self._resolve_pretrained_source()
        AutoModelForCausalLM = get_hub_class(self.model_hub, "AutoModelForCausalLM")
        logger.info(
            f"Loading full model from: '{source}' with kwargs: {self.model_kwargs}"
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path=source,
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
        bh.empty_cache()

    def forward(self, *args, **kwargs):
        if self.model is None:
            raise RuntimeError("Forward requires full model runtime. Call prepare_full_model().")
        return self.model(*args, **kwargs)
