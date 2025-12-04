from typing import Type, Optional, Dict
import os
import pkgutil
import importlib

class BaseFactory:
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
            raise KeyError(f"'{name}' not found in {cls.__name__}. Available: {available}")
        return cls._registry[key]

    @classmethod
    def create(cls, *args, **kwargs):
        if "type" not in kwargs:
            raise ValueError("Missing required 'type' in kwargs")
        name = kwargs.pop("type")
        target_cls = cls.get(name)
        return target_cls(*args, **kwargs)


    @classmethod
    def available(cls):
        cls._lazy_import()
        return list(cls._registry.keys())

    
class ModelFactory(BaseFactory):
    _registry: Dict[str, Type] = {}

MODELS_PACKAGE = "npuslim.model"
MODELS_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "model")
ModelFactory.set_package(MODELS_PACKAGE, MODELS_PATH)

class DatasetFactory(BaseFactory):
    _registry: Dict[str, Type] = {}

DATALOADER_PACKAGE = "npuslim.dataset"
DATALOADER_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "dataset")
DatasetFactory.set_package(DATALOADER_PACKAGE, DATALOADER_PATH)

class QuantFactory(BaseFactory):
    _registry: Dict[str, Type] = {}

QUANTS_PACKAGE = "npuslim.quant"
QUANTS_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "quant")
QuantFactory.set_package(QUANTS_PACKAGE, QUANTS_PATH)






