from safetensors import safe_open

file_path = "/home/lichangcai/projects/llm/npuslim/outputs/qwen3-32b_int8_dyn/model-00001-of-00014.safetensors"

with safe_open(file_path, framework="pt") as f:
    # 获取所有张量的名称
    keys = f.keys()
    # 取出第一个张量查看其数据类型
    first_key = list(keys)[11]
    tensor_info = f.get_slice(first_key)
    
    print(f"张量名称: {first_key}")
    print(f"数据类型 (dtype): {f.get_tensor(first_key).dtype}")