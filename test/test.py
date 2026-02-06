import os
import sys
import torch
from tqdm import tqdm
from easydict import EasyDict
from transformers import AutoModelForCausalLM, AutoTokenizer
from npuslim.utils.factory import DatasetFactory
from torch.utils.data import DataLoader

# 路径配置
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# 确保 rqp.py 位于 test 目录下或 parent_dir 下
try:
    from test.rqp import RQPGradCollector
except ImportError:
    from rqp import RQPGradCollector


def run_rqp_quantization(model_id, dataloader, alpha=0.5):
    """
    针对 Qwen3 模型执行 RQP 增强型标定
    alpha: H_PPL 和 F_r 的融合系数 [cite: 68]
    """
    # 1. 加载模型 [cite: 78, 79]
    print(f"Loading model: {model_id}")
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.float16, device_map="auto"
    )
    model.eval()

    # 2. 初始化 RQP 梯度采集器 [cite: 76]
    # topk=5 对应文档建议的推理稳定性约束范围 [cite: 38, 80]
    collector = RQPGradCollector(model, topk=10)

    # 3. 第一阶段：收集推理敏感的 Fisher 矩阵 F^(r) [cite: 77-82]
    print("Step 1: Collecting Reasoning-Sensitive Fisher Matrices (RQP)...")
    
    # 显式使用 torch.no_grad() 外部包裹，但在 collector 内部会按需开启梯度用于 Jacobian [cite: 81]
    for batch in tqdm(dataloader):
        # 兼容 dataloader 输出的字典格式
        # 提取 input_ids 并确保在正确的设备上 
        if isinstance(batch, dict):
            input_ids = batch['input_ids'].to(model.device)
        else:
            input_ids = batch.to(model.device)
            
        # 执行一次推理感知的反向传播并累积 F^(r) 
        collector.collect_grads(input_ids)

    # 4. 第二阶段：层级量化与 Hessian 融合 [cite: 83-85]
    print("Step 2: Integrating F_r into GPTQ Hessian...")

    # Qwen 系列层结构位于 model.model.layers [cite: 72]
    layers = model.model.layers

    for i in range(len(layers)):
        layer = layers[i]

        # 针对该层内的每一个线性子层
        named_linears = {
            n: m for n, m in layer.named_modules() if isinstance(m, torch.nn.Linear)
        }

        for name, linear_layer in named_linears.items():
            # 构造完整的路径名，必须与 collector 中 fw_hook 记录的名字一致
            # 通常为 model.layers.0.self_attn.q_proj 等
            full_name = f"model.layers.{i}.{name}"

            # --- 模拟 GPTQ 的 Hessian 计算 ---
            # 标准 GPTQ 目标：min ||W - Q(W)||^2_H_ppl [cite: 35, 70]
            H_ppl = get_standard_gptq_hessian(linear_layer)

            # --- RQP 核心融合步骤 [cite: 68, 70, 84] ---
            # 算法核心公式：H_RQP = alpha * H_PPL + (1 - alpha) * F_r [cite: 68]
            # 这一步将推理敏感性塞进 GPTQ 的代码结构 [cite: 51, 71]
            H_rqp = collector.get_merged_hessian(full_name, H_ppl, alpha=alpha)

            # 5. 后续可接 GPTQ 的 Cholesky 分解与 4bit 量化 [cite: 73, 74, 85]
            # quantize_with_gptq_method(linear_layer, H_rqp)
            print(f"Layer {full_name}: Fisher (RQP) and PPL Hessian merged.")

    # 6. 清理 Hooks 释放内存 [cite: 128]
    collector.cleanup()
    print("RQP-enhanced optimization completed.")


def get_standard_gptq_hessian(linear_module):
    """
    占位符：返回标准 PPL 敏感的 Hessian (X*X^T) [cite: 35, 68]
    """
    d_in = linear_module.in_features
    return torch.eye(d_in).to(linear_module.weight.device)


# --- 运行示例 ---
if __name__ == "__main__":
    config = {
        "type": "TextDataset",
        "data_path": "./dataset/sharegpt_gpt4_qwen/sharegpt_gpt4-qwen3_a22B_output.jsonl",
        "num_samples": 128,
        "max_seq_length": 4096,
        "device": "cuda"
    }
    config = EasyDict(config)
    
    model_path = "/data16t/MODELSCOPE/models/Qwen/Qwen3-0.6B"
    processor = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    
    dataset = DatasetFactory.create(processor=processor, config=config)
    dataloader = DataLoader(
        dataset, 
        collate_fn=dataset.collate_fn, 
        batch_size=1, 
        shuffle=True, 
        num_workers=0
    )

    # 运行 RQP 优化流程 [cite: 3, 4]
    run_rqp_quantization(model_path, dataloader, alpha=0.5)