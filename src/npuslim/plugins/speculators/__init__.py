"""NPUSlim Speculators Plugin.

Registers NPUSlim compatibility patches for the external ``speculators``
package without vendoring or forking upstream code.
"""

from __future__ import annotations

from pathlib import Path

from loguru import logger


def register():
    """Register NPUSlim extensions with speculators.

    The actual patches are discovered lazily from the local
    ``npuslim.plugins.speculators`` package so future overrides can be added
    without changing the top-level bootstrap flow.
    """
    try:
        from npuslim.plugins.registry import apply_all_patches, discover_modules

        plugin_dir = str(Path(__file__).parent)
        discover_modules("npuslim.plugins.speculators", plugin_dir)
        apply_all_patches()
        logger.info("NPUSlim registered with speculators")
    except ImportError as e:
        logger.warning(f"Could not register NPUSlim with speculators: {e}")
