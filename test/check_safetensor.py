import torch
from safetensors import safe_open
import os

def debug_safetensors(file_path, limit=50):
    """
    查看 safetensor 内容：键名、形状、数据类型及数值预览
    :param file_path: .safetensors 文件路径
    :param limit: 最多查看多少个 key，防止控制台刷屏
    """
    if not os.path.exists(file_path):
        print(f"❌ 文件不存在: {file_path}")
        return

    print(f"\n{'='*120}")
    print(f"📁 文件: {os.path.basename(file_path)}")
    print(f"{'Key':<55} | {'Dtype':<15} | {'Shape':<15} | {'Data Preview (First 3)'}")
    print(f"{'-'*120}")

    with safe_open(file_path, framework="pt", device="cpu") as f:
        keys = sorted(f.keys())
        for i, key in enumerate(keys):
            if i >= limit:
                print(f"... 还有 {len(keys) - limit} 个参数未显示")
                break
            
            # 获取真实的 tensor 数据
            tensor = f.get_tensor(key)
            
            # 格式化数据预览：取前 5 个并转为 list 方便打印
            # flatten 是为了处理多维矩阵也能看到开头
            preview_data = tensor.flatten()[:3].tolist()
            preview_str = ", ".join([f"{v:.4f}" if isinstance(v, float) else str(v) for v in preview_data])
            
            print(f"{key[:55]:<55} | {str(tensor.dtype):<15} | {str(list(tensor.shape)):<15} | [{preview_str}...]")

    print(f"{'='*120}\n")

# 使用示例
debug_safetensors("outputs/qwen3-30b_a3b_int8_dyn/model-00001-of-00013.safetensors")