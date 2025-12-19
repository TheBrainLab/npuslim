


def register():
    # 局部导入，避免循环引用和过早导入
    try:
        from vllm.model_executor.layers.quantization import QUANTIZATION_METHODS
        from .quantization.adapter import NpuSlimConfig
        
        # 1. 注册量化配置类
        method_name = "npuslim"
        if method_name not in QUANTIZATION_METHODS:
            QUANTIZATION_METHODS[method_name] = NpuSlimConfig
            print(f">>> [NpuSlim] Quantization '{method_name}' registered successfully.")
        
        # 2. 如果你未来还要注册新模型架构，也可以写在这里
        # from vllm import ModelRegistry
        # if "MyNewModel" not in ModelRegistry.get_supported_archs():
        #     ModelRegistry.register_model(...)
            
    except Exception as e:
        print(f">>> [NpuSlim] Registration failed: {e}")