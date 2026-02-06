from typing import Type, Optional, Dict, Any, TYPE_CHECKING
from abc import ABC, abstractmethod
import importlib
from loguru import logger

if TYPE_CHECKING:
    from npuslim.utils.config_parser import ModelConfig, DatasetConfig

# ================================= Global Settings ================================= #
ROOT_PACKAGE = "npuslim"


class BaseFactory(ABC):
    _registry: Dict[str, Type] = {}
    _map_loaded: bool = False
    _SUBMODULE: Optional[str] = None
    _lazy_map: Dict[str, str] = {}

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls._registry = {}
        cls._lazy_map = {}
        cls._map_loaded = False

    @classmethod
    def _load_submodule_map(cls):
        if cls._map_loaded or not cls._SUBMODULE:
            return

        package_name = f"{ROOT_PACKAGE}.{cls._SUBMODULE}"
        try:
            pkg = importlib.import_module(package_name)
            if hasattr(pkg, "_REGISTRY_MAP"):
                raw_map = getattr(pkg, "_REGISTRY_MAP")
                if not isinstance(raw_map, dict):
                    logger.warning(
                        f"⚠️ [Factory] _REGISTRY_MAP in {package_name} is not a dict."
                    )
                    return

                for algo_name, rel_path in raw_map.items():
                    key = algo_name.lower()
                    if rel_path.startswith("."):
                        full_path = f"{package_name}{rel_path}"
                    else:
                        full_path = rel_path
                    cls._lazy_map[key] = full_path
            else:
                logger.debug(
                    f"ℹ️ [Factory] No _REGISTRY_MAP found in {package_name}. Only pre-imported classes are available."
                )

        except ImportError as e:
            logger.error(
                f"❌ [Factory] Critical: Could not import package '{package_name}': {e}"
            )

        cls._map_loaded = True

    @classmethod
    def register(cls, name: Optional[str] = None):
        def decorator(target_cls: Type):
            reg_name = (name or target_cls.__name__).lower()
            if reg_name in cls._registry:
                raise KeyError(f"'{reg_name}' already registered in {cls.__name__}")
            cls._registry[reg_name] = target_cls
            return target_cls

        return decorator

    @classmethod
    def get(cls, name: str) -> Type:
        key = name.lower()
        if key in cls._registry:
            return cls._registry[key]

        cls._load_submodule_map()

        if key in cls._lazy_map:
            module_path = cls._lazy_map[key]
            try:
                # logger.debug(f"🎯 [Factory] Loading: {key} -> {module_path}")
                importlib.import_module(module_path)
                if key in cls._registry:
                    return cls._registry[key]
                else:
                    raise ImportError(
                        f"Module '{module_path}' was imported, but '{key}' was not found in registry. "
                        f"Did you forget @{cls.__name__}.register or match the name?"
                    )
            except Exception as e:
                logger.error(
                    f"❌ [Factory] Failed to import '{module_path}' for '{key}': {e}"
                )
                raise e

        available = list(cls._registry.keys()) + list(cls._lazy_map.keys())
        msg = f"'{name}' not found in {cls.__name__}."
        if not available:
            msg += " Registry is empty. Did you define _REGISTRY_MAP in __init__.py?"
        else:
            msg += f" Available: {', '.join(sorted(available))}"

        raise KeyError(msg)

    @classmethod
    @abstractmethod
    def create(cls, *args, **kwargs): ...


# ================================= Concrete Factories ================================= #


class ModelFactory(BaseFactory):
    _SUBMODULE = "model"

    @classmethod
    def create(cls, *args, config: "ModelConfig", **kwargs):
        return cls.get(config.type)(*args, config=config, **kwargs)


class DatasetFactory(BaseFactory):
    _SUBMODULE = "dataset"

    @classmethod
    def create(cls, *args, config: "DatasetConfig", **kwargs):
        return cls.get(config.type)(*args, config=config, **kwargs)


class TaskFactory(BaseFactory):
    _SUBMODULE = "tasks"

    @classmethod
    def create(
        cls, task_key: str, raw_config: Dict[str, Any], resources: Dict[str, Any]
    ):
        return cls.get(task_key)(config=raw_config, resources=resources)


class CompressorFactory(BaseFactory):
    _SUBMODULE = "compressor"

    @classmethod
    def create(cls, algo_name: str, *args, **kwargs):
        return cls.get(algo_name)(*args, **kwargs)


class SaverFactory(BaseFactory):
    _SUBMODULE = "saver"

    @classmethod
    def create(cls, format_name: str, *args, **kwargs):
        return cls.get(format_name)(*args, **kwargs)
