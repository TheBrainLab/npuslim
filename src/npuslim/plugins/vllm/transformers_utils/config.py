"""Patch vllm.transformers_utils.config for NPUSlim custom model types."""

from vllm.logger import init_logger

from npuslim.plugins.registry import register_patch

logger = init_logger(__name__)


@register_patch("vllm.transformers_utils.config")
def patch_vllm_config_registry(module):
    """Register NPUSlim config aliases in vLLM config registry."""
    # Map the custom model_type to the same base config class used by kimi_k2.
    # This keeps compatibility with existing fields while giving us an isolated
    # runtime model_type for plugin routing.
    if module._CONFIG_REGISTRY.get("kimi_k2_mcore") != "DeepseekV3Config":
        module._CONFIG_REGISTRY["kimi_k2_mcore"] = "DeepseekV3Config"
        logger.info("Registered vLLM config alias: kimi_k2_mcore -> DeepseekV3Config")
