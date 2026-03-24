from npuslim.registry import ModelRegistry

ModelRegistry.register_lazy("Qwen3", ".qwen3", aliases=["Qwen3Model"])
ModelRegistry.register_lazy("OPT", ".opt", aliases=["OPTModel"])
