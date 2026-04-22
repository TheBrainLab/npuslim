"""Patch vllm.transformers_utils.model_arch_config_convertor for NPUSlim models."""

from vllm.logger import init_logger

from npuslim.plugins.registry import register_patch

logger = init_logger(__name__)


@register_patch("vllm.transformers_utils.model_arch_config_convertor")
def patch_model_arch_convertors(module):
    """Register ModelArchitectureConfig convertor for kimi_k2_mcore."""

    class KimiK2MCoreModelArchConfigConvertor(module.ModelArchConfigConvertorBase):
        """Convertor for NPUSlim Kimi-K2 MCore converted checkpoints."""

        def get_head_size(self) -> int:
            # NPUSlim MCore checkpoints use kv_channels as attention head dim.
            kv_channels = getattr(self.hf_text_config, "kv_channels", None)
            if kv_channels is not None:
                return kv_channels
            return super().get_head_size()

        def get_total_num_kv_heads(self) -> int:
            # qkv projection is grouped-query style with `num_query_groups`.
            num_query_groups = getattr(self.hf_text_config, "num_query_groups", None)
            if num_query_groups is not None:
                return num_query_groups
            return super().get_total_num_kv_heads()

        def is_deepseek_mla(self) -> bool:
            # NPUSlim MCore flavor is q/k/v projection based (non-MLA).
            return False

    module.MODEL_ARCH_CONFIG_CONVERTORS["kimi_k2_mcore"] = (
        KimiK2MCoreModelArchConfigConvertor
    )
    logger.info("Registered ModelArchitectureConfig convertor for kimi_k2_mcore")
