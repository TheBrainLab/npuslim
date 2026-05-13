from __future__ import annotations

import importlib
from abc import ABC
from pathlib import Path
from typing import Any, Dict, List, Optional

from accelerate import init_empty_weights
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
            should_pop = False
            if key in passthrough_model_keys and key not in merged_model_kwargs:
                merged_model_kwargs[key] = kwargs[key]
                should_pop = True
            if key in passthrough_tokenizer_keys and key not in merged_tokenizer_kwargs:
                merged_tokenizer_kwargs[key] = kwargs[key]
                should_pop = True
            if should_pop:
                kwargs.pop(key)

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
        self.empty_model = None
        self.tokenizer = None
        self.processor = None
        self.config = None
        self.quantized = False
        self.model_type = "LLM"

        self.prepare_metadata()

    def get_model_loader_candidates(self) -> List[str]:
        """Ordered model loader class candidates."""
        return ["AutoModelForCausalLM"]

    def get_tokenizer_loader_candidates(self) -> List[str]:
        """Ordered tokenizer loader class candidates."""
        return ["AutoTokenizer"]

    def get_processor_loader_candidates(self) -> List[str]:
        """Ordered processor loader class candidates. Empty means no processor."""
        return []

    def _resolve_first_available_class(
        self,
        class_names: List[str],
        *,
        kind: str,
    ):
        if not class_names:
            raise RuntimeError(f"No {kind} class candidates provided")

        errors: List[str] = []
        for class_name in class_names:
            try:
                return get_hub_class(self.model_hub, class_name)
            except Exception as exc:
                errors.append(f"{class_name}: {exc}")

        raise RuntimeError(
            f"Failed to resolve {kind} class. "
            f"Candidates={class_names}. Errors={errors}"
        )

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
        tokenizer_cls = self._resolve_first_available_class(
            self.get_tokenizer_loader_candidates(),
            kind="tokenizer",
        )

        logger.info(
            f"Loading tokenizer metadata from: '{source}' with kwargs: {self.tokenizer_kwargs}"
        )
        self.tokenizer = tokenizer_cls.from_pretrained(
            pretrained_model_name_or_path=source,
            **self.tokenizer_kwargs,
        )

        processor_candidates = self.get_processor_loader_candidates()
        if processor_candidates:
            try:
                processor_cls = self._resolve_first_available_class(
                    processor_candidates,
                    kind="processor",
                )
                self.processor = processor_cls.from_pretrained(
                    pretrained_model_name_or_path=source,
                    **self.tokenizer_kwargs,
                )
            except Exception as exc:
                logger.warning(f"Failed to load processor metadata: {exc}")
                self.processor = None

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
        model_cls = self._resolve_first_available_class(
            self.get_model_loader_candidates(),
            kind="model",
        )
        logger.info(
            f"Loading full model from: '{source}' with kwargs: {self.model_kwargs}"
        )
        self.model = model_cls.from_pretrained(
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

    def prepare_empty_model(self):
        """Build and cache a meta-device model skeleton without loading weights."""
        if self.empty_model is not None:
            return self.empty_model

        if self.config is None:
            self.prepare_metadata()
        if self.config is None:
            raise RuntimeError("Config is required to build empty model skeleton.")

        model_cls = self._resolve_first_available_class(
            self.get_model_loader_candidates(),
            kind="model",
        )
        trust_remote_code = bool(self.model_kwargs.get("trust_remote_code", False))
        extra_kwargs: Dict[str, Any] = {}
        if "attn_implementation" in self.model_kwargs:
            extra_kwargs["attn_implementation"] = self.model_kwargs["attn_implementation"]
        if "torch_dtype" in self.model_kwargs:
            extra_kwargs["torch_dtype"] = self.model_kwargs["torch_dtype"]

        with init_empty_weights():
            self.empty_model = model_cls.from_config(
                self.config,
                trust_remote_code=trust_remote_code,
                **extra_kwargs,
            )
        self.empty_model.eval()
        return self.empty_model

    def release_empty_model(self) -> None:
        self.empty_model = None
        bh.empty_cache()

    def forward(self, *args, **kwargs):
        if self.model is None:
            raise RuntimeError("Forward requires full model runtime. Call prepare_full_model().")
        return self.model(*args, **kwargs)

    def adapt_gptq_runtime_model(self, runtime_model):
        """Optional model-specific GPTQ runtime adaptation hook."""
        return runtime_model
