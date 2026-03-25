# src/npuslim/core/resource_config.py
"""Resource configuration - standalone to avoid circular imports."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict


@dataclass
class MetadataConfig:
    """Top-level metadata configuration."""

    name: str = ""
    description: str = ""


@dataclass
class ResourceConfig:
    """Resource declaration."""

    id: str
    type: str
    extra: Dict[str, Any] = field(default_factory=dict)
