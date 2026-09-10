"""QAD Phase 1 训练通路（M1）。

模块：
- ste:      STE fake-quant（forward 精确复刻部署量化网格，backward 恒等直通）
- kl_loss:  chunked KL 蒸馏 loss（top-K teacher + 质量守恒均摊，防 fp32 log_softmax OOM）
- lora:     per-expert LoRA（output-space，delta 加在 fake-quant 权重之后）
- tiny_moe: tiny MoE 验证模型（M3 sanity / A7 消融共用，无 HF 依赖）

约定（与部署语义对齐，见 W4A8_修改总结.md §10.9）：
- W4A8 专家权重：对称 int4 per-output-channel，q ∈ [-8,7]（补码 nibble 存储），
  有效权重 = q * scale（scale_bias 是 kernel 侧激活表征补偿，不进入 float 训练路径）
- W8A8 稠密权重：对称 int8 per-output-channel，q ∈ [-128,127]
"""
