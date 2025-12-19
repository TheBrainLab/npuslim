from typing import Type, Optional, Dict
from abc import ABC, abstractmethod
import pkgutil
import importlib
from pathlib import Path

# from dataclasses import is_dataclass, asdict
from .config_parser import ModelConfig, CalibDatasetConfig, CompressorConfig



class BaseFactory(ABC):
    """Base registry factory supporting subclass registries with lazy import."""

    _registry: Dict[str, Type] = {}
    _package: Optional[str] = None  # Package path for lazy import
    _package_dir: Optional[str] = None

    @classmethod
    def set_package(cls, package: str, package_dir: str):
        """Set the package to lazy import for automatic registration."""
        cls._package = package
        cls._package_dir = package_dir

    @classmethod
    def _lazy_import(cls):
        if cls._package and cls._package_dir:
            for finder, name, ispkg in pkgutil.walk_packages([cls._package_dir]):
                if name.startswith("_"):
                    continue
                importlib.import_module(f"{cls._package}.{name}")
            cls._package = None  # avoid repeated imports

    @classmethod
    def register(cls, name: Optional[str] = None):
        """Decorator for registering a class."""

        def decorator(target_cls: Type):
            reg_name = (name or target_cls.__name__).lower()
            if reg_name in cls._registry:
                raise KeyError(
                    f"'{reg_name}' already registered in {cls.__name__} "
                    f"by {cls._registry[reg_name].__name__}"
                )
            cls._registry[reg_name] = target_cls
            return target_cls

        return decorator

    @classmethod
    def get(cls, name: str) -> Type:
        cls._lazy_import()
        key = name.lower()
        if key not in cls._registry:
            available = ", ".join(cls._registry.keys())
            raise KeyError(
                f"'{name}' not found in {cls.__name__}. Available: {available}"
            )
        return cls._registry[key]

    @classmethod
    @abstractmethod
    def create(cls, *args, **kwargs): ...

    @classmethod
    def available(cls):
        cls._lazy_import()
        return list(cls._registry.keys())


class ModelFactory(BaseFactory):
    _registry: Dict[str, Type] = {}

    @classmethod
    def create(cls, *args, config: "ModelConfig", **kwargs):
        name = config.type
        target_cls = cls.get(name)
        return target_cls(*args, config=config, **kwargs)


MODELS_PACKAGE = "npuslim.model"
MODELS_PATH = Path(__file__).resolve().parent.parent / "model"
ModelFactory.set_package(MODELS_PACKAGE, str(MODELS_PATH))


class DatasetFactory(BaseFactory):
    _registry: Dict[str, Type] = {}

    @classmethod
    def create(cls, *args, config: "CalibDatasetConfig", **kwargs):
        name = config.type
        target_cls = cls.get(name)
        return target_cls(*args, config=config, **kwargs)


DATALOADER_PACKAGE = "npuslim.dataset"
DATALOADER_PATH = Path(__file__).resolve().parent.parent / "dataset"
DatasetFactory.set_package(DATALOADER_PACKAGE, str(DATALOADER_PATH))


class CompressorFactory(BaseFactory):
    _registry: Dict[str, Type] = {}

    @classmethod
    def create(cls, *args, config: "CompressorConfig", **kwargs):
        name = config.type
        target_cls = cls.get(name)
        return target_cls(*args, **kwargs)


COMPRESSOR_PACKAGE = "npuslim.compressor"
COMPRESSOR_PATH = Path(__file__).resolve().parent.parent / "compressor"
CompressorFactory.set_package(COMPRESSOR_PACKAGE, str(COMPRESSOR_PATH))
