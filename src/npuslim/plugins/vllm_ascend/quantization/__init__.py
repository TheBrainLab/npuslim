"""Quantization extensions for vLLM-Ascend.

This package contains:
- Patches for vllm-ascend quantization modules
- New quantization schemes (W4A16, etc.)

All extensions use decorators for self-registration:
- @register_patch from npuslim.plugins.registry for patches
- @register_scheme from vllm-ascend for schemes
"""
