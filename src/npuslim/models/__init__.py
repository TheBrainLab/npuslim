from npuslim.registry import ModelRegistry

ModelRegistry.register_lazy("Qwen3", ".qwen3", aliases=["Qwen3Model"])
ModelRegistry.register_lazy(
    "Qwen3VL",
    ".qwen3_vl",
    aliases=["Qwen3VLModel", "Qwen3VLMoe", "Qwen3VLMoeModel"],
)
ModelRegistry.register_lazy("OPT", ".opt", aliases=["OPTModel"])
ModelRegistry.register_lazy("GLM5", ".glm5", aliases=["Glm5Model", "GlmMoeDsa"])
