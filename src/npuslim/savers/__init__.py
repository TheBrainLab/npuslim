# src/npuslim/savers/__init__.py
"""Model savers for NPUSlim."""

from npuslim.registry import SaverRegistry

# Lazy registration for savers
SaverRegistry.register_lazy(
    "StreamingHuggingFaceSaver",
    ".hf_saver",
    aliases=["streaming_hf", "StreamingHFSaver", "streaming_hf_saver"],
)
