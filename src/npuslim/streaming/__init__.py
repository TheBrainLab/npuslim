"""Streaming utilities for NPUSlim.

DEPRECATED: Use npuslim.tasks.compressor.loader and npuslim.savers instead.
"""

import warnings

warnings.warn(
    "npuslim.streaming is deprecated. "
    "Use npuslim.tasks.compressor.loader and npuslim.savers instead.",
    DeprecationWarning,
    stacklevel=2,
)

# Keep backward compatibility - import from new locations
from npuslim.tasks.compressor.loader import ChunkLoader as StreamLoader
from npuslim.savers.hf_saver import HuggingFaceSaver as StreamSaver

# Keep old names for compatibility
from npuslim.streaming.streaming import (
    SafeTensorIndex,
    SafeTensorStreamLoader,
    ShardTensorReader,
)

__all__ = [
    "SafeTensorIndex",
    "ShardTensorReader",
    "StreamLoader",
    "SafeTensorStreamLoader",
    "StreamSaver",
    "ChunkLoader",
    "HuggingFaceSaver",
]
