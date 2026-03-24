import importlib
from typing import Any, Dict, List, Optional, Type

ROOT_PACKAGE = "npuslim"


class Registry:
    """Auto-discovery lazy registry with submodule support."""

    def __init__(self, name: str, submodule: Optional[str] = None):
        self.name = name
        self._submodule = submodule
        self._registry: Dict[str, Type] = {}
        self._aliases: Dict[str, str] = {}
        self._lazy_map: Dict[str, str] = {}
        self._submodule_loaded = False

    def _ensure_submodule_loaded(self) -> None:
        """Auto-import submodule's __init__.py on first access."""
        if self._submodule_loaded or not self._submodule:
            return
        try:
            importlib.import_module(f"{ROOT_PACKAGE}.{self._submodule}")
        except ImportError:
            pass
        self._submodule_loaded = True

    def register(self, name, aliases: Optional[List[str]] = None):
        """
        Decorator: @DatasetRegistry.register() or @DatasetRegistry.register("c4", aliases=["C4"])
        """
        def decorator(cls: Type) -> Type:
            self._do_register(name, cls, aliases)
            return cls
        return decorator

    def _do_register(self, name: str, cls: Type, aliases: Optional[List[str]] = None):
        name = name.lower()
        if name in self._registry:
            return
        self._registry[name] = cls
        if aliases:
            for alias in aliases:
                self._aliases[alias.lower()] = name

    def register_lazy(self, name: str, module_path: str, aliases: Optional[List[str]] = None):
        """Register module path for lazy loading. Supports relative paths like ".c4_dataset"."""
        name = name.lower()
        # Convert relative path: ".c4_dataset" -> "npuslim.datasets.c4_dataset"
        if module_path.startswith("."):
            if self._submodule:
                module_path = f"{ROOT_PACKAGE}.{self._submodule}{module_path}"
            else:
                raise ValueError(f"Registry '{self.name}' has no submodule, cannot use relative path")
        self._lazy_map[name] = module_path
        if aliases:
            for alias in aliases:
                self._aliases[alias.lower()] = name

    def get(self, name: str) -> Type:
        """Get class by name (with lazy loading)."""
        key = name.lower()
        resolved = self._aliases.get(key, key)

        # Check eager registry
        if resolved in self._registry:
            return self._registry[resolved]

        # Ensure submodule loaded (triggers register_lazy calls in __init__.py)
        self._ensure_submodule_loaded()

        # Lazy loading
        if resolved in self._lazy_map:
            importlib.import_module(self._lazy_map[resolved])
            if resolved in self._registry:
                return self._registry[resolved]

        available = sorted(set(self._registry.keys()) | set(self._lazy_map.keys()))
        raise KeyError(f"{self.name} '{name}' not found. Available: {available}")

    def create(self, name: str, *args, **kwargs) -> Any:
        """Create instance by name."""
        return self.get(name)(*args, **kwargs)

    def list(self) -> List[str]:
        """List all registered names (including lazy)."""
        self._ensure_submodule_loaded()
        return sorted(set(self._registry.keys()) | set(self._lazy_map.keys()))

    def clear(self) -> None:
        """Clear all registrations (for testing)."""
        self._registry.clear()
        self._aliases.clear()
        self._lazy_map.clear()
        self._submodule_loaded = False


# === Global singletons ===
ModelRegistry = Registry("Model", "models")
DatasetRegistry = Registry("Dataset", "datasets")
TaskRegistry = Registry("Task", "tasks")
SaverRegistry = Registry("Saver", "savers")
