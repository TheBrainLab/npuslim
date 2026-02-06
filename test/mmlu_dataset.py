import torch
import random
import numpy as np
from datasets import load_dataset

def get_mmlu_random_calib_data(n_samples=128, seed=42):
    """
    随机采集 MMLU 数据并固定随机数。
    """
    # 1. 固定所有随机种子 [cite: 3]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    print(f"Loading MMLU validation set and sampling {n_samples} items...")
    
    # 2. 加载 MMLU 验证集 (包含 57 个学科，确保多样性) 
    # 使用 'all' 载入全集，并根据需要处理报错 [cite: 3]
    try:
        dataset = load_dataset("cais/mmlu", "all", split="validation")
    except Exception as e:
        print(f"Error loading dataset: {e}. Please check your 'datasets' version.")
        return []

    # 3. 随机选择索引 
    total_len = len(dataset)
    if n_samples > total_len:
        n_samples = total_len
    
    indices = random.sample(range(total_len), n_samples)
    
    formatted_prompts = []
    choices_map = ["A", "B", "C", "D"]

    # 4. 转化为 Reasoning-style Prompt [cite: 6, 78]
    for idx in indices:
        item = dataset[idx]
        question = item['question']
        choices = item['choices']
        
        # 构造符合推理逻辑的输入格式 [cite: 6, 8]
        prompt = f"The following are multiple choice questions (with answers).\n\n"
        prompt += f"Question: {question}\n"
        for i, choice in enumerate(choices):
            prompt += f"{choices_map[i]}. {choice}\n"
        prompt += "Answer:"
        
        formatted_prompts.append(prompt)

    return formatted_prompts

# --- 使用示例 ---
# 采集 128 条固定随机的数据 [cite: 102]
calib_prompts = get_mmlu_random_calib_data(n_samples=128, seed=42)

# 打印第一条数据检查格式 [cite: 79]
print("\n--- First Sample ---")
print(calib_prompts[0])
print('######################')