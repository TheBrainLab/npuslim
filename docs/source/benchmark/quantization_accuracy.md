# Quantization Accuracy

[TO BE ADDED]

## Overview

Comparison of quantization accuracy across different algorithms and bit widths.

## Results

| Model | Algorithm | Bits | PPL (WikiText2) | PPL (C4) |
|--------|-----------|-------|-------------------|-------------|
| Qwen3-0.6B | FP16 | 16 | - | - |
| Qwen3-0.6B | INT8Dynamic | 8 | - | - |
| Qwen3-0.6B | GPTQ | 4 | - | - |
| Qwen3-0.6B | QuIP | 4 | - | - |

## Evaluation Method

```bash
# WikiText2 perplexity
python tools/eval_ppl_quip_style.py --model outputs/qwen3_int8_dyn

# C4 perplexity
python tools/eval_ppl_quip_style.py --model outputs/qwen3_int8_dyn --dataset c4
```

## Analysis

[Detailed analysis of accuracy vs compression trade-offs]
