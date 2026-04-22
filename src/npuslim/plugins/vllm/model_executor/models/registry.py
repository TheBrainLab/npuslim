"""Patch vllm.model_executor.models.registry for NPUSlim custom architectures."""

from vllm.logger import init_logger

from npuslim.plugins.registry import register_patch

logger = init_logger(__name__)


@register_patch("vllm.model_executor.models.registry")
def patch_model_registry(module):
    """Register NPUSlim architecture into vLLM model registry."""
    model_arch = "KimiK2MCoreForCausalLM"

    module.ModelRegistry.register_model(
        model_arch,
        "npuslim.plugins.vllm.model_executor.models.kimi_k2_mcore:"
        "KimiK2MCoreForCausalLM",
    )
    logger.info("Registered custom architecture %s in ModelRegistry", model_arch)
