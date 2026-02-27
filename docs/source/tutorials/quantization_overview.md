# Quantization Methods Overview

NPUSlim supports multiple post-training quantization (PTQ) algorithms. This page provides a comparison to help you choose the right method for your use case.

## Algorithm Comparison

| Algorithm | Weight Bits | Activation | Speed | Accuracy | Best For |
|-----------|-------------|------------|-------|----------|----------|
| INT8Dynamic | 8 | Dynamic INT8 | ⚡⚡⚡ | Good | General purpose, fast |
| GPTQ | 4-8 | FP16 | ⚡⚡ | Better | Accuracy-critical tasks |
| QuIP | 2-4 | FP16 | ⚡ | Best | Extreme compression |
| SparseGPT | 4-8 + Pruning | FP16 | ⚡⚡ | Good | Structured sparsity |

## When to Use Each

### INT8Dynamic

- **Recommended for**: First-time users, production deployment
- **Pros**: Fast quantization, no calibration data needed
- **Cons**: Lower compression ratio

### GPTQ

- **Recommended for**: When accuracy matters more than speed
- **Pros**: Better accuracy at lower bits
- **Cons**: Slower quantization, requires calibration data

### QuIP

- **Recommended for**: Extreme compression (2-4 bits)
- **Pros**: State-of-the-art low-bit quantization
- **Cons**: Slowest quantization, experimental

### SparseGPT

- **Recommended for**: When you need both pruning and quantization
- **Pros**: Combines sparsity with quantization
- **Cons**: More complex, requires tuning

## Quick Reference

```bash
# INT8Dynamic
python tools/run.py -c configs/compressor/int8_dyn/qwen3_0_6b.yaml

# GPTQ
python tools/run.py -c configs/compressor/gptq/qwen3_0_6b.yaml

# QuIP
python tools/run.py -c configs/compressor/quip/qwen3_0_6b-w4.yaml

# SparseGPT
python tools/run.py -c configs/compressor/sparsegpt/qwen3_0_6b.yaml
```
