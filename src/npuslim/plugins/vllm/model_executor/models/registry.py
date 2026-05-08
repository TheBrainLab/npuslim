"""Patch vllm.model_executor.models.registry for NPUSlim custom architectures."""

from vllm.logger import init_logger

from npuslim.plugins.registry import register_patch

logger = init_logger(__name__)


@register_patch("vllm.model_executor.models.registry")
def patch_model_registry(module):
    """Register NPUSlim architectures into vLLM model registry."""
    arch_to_entrypoint = {
        "KimiK2MCoreForCausalLM": (
            "npuslim.plugins.vllm.model_executor.models.kimi_k2_mcore:"
            "KimiK2MCoreForCausalLM"
        ),
        "KimiK2MCoreV2ForCausalLM": (
            "npuslim.plugins.vllm.model_executor.models.kimi_k2_mcore_v2:"
            "KimiK2MCoreV2ForCausalLM"
        ),
        "KimiK2MCoreV1ForCausalLM": (
            "npuslim.plugins.vllm.model_executor.models.kimi_k2_mcore_v1:"
            "KimiK2MCoreV1ForCausalLM"
        ),
    }

    for model_arch, entrypoint in arch_to_entrypoint.items():
        module.ModelRegistry.register_model(model_arch, entrypoint)
        logger.info("Registered custom architecture %s in ModelRegistry", model_arch)
