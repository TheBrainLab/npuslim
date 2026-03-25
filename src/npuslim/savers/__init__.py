# src/npuslim/savers/__init__.py
"""Model savers for NPUSlim."""

from npuslim.registry import SaverRegistry

# Lazy registration for savers
SaverRegistry.register_lazy("HuggingFaceSaver",".hf_saver", aliases=["HF"])
