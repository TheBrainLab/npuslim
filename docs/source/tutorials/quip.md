# QuIP Quantization

QuIP (Quaternion-Inspired Post-training Quantization) is a high-quality 4-bit quantization method using LDLQ decomposition.

## Overview

- **Weight Quantization**: 4-bit, no grouping
- **Algorithm**: LDLQ (Low-Distortion Linear Quantization)
- **Vector Balancing**: Optimizes quantization grid
- **Two Modes**: MinMax and RMS quantization functions

## Configuration

```yaml
pipeline:
  - type: ptq
    algo_name: QuIP
    ignore_layers: []
    algo_config:
      w_bits: 4
      quant_func: rms        # or minmax
      vector_balance_iters: 20
    calib_config:
      num_samples: 256
```

### Parameters

| Parameter | Type | Description | Default |
|-----------|--------|-------------|----------|
| `w_bits` | int | Weight bit width (2, 3, 4) | 4 |
| `quant_func` | str | Quantization function: `rms` or `minmax` | rms |
| `vector_balance_iters` | int | Vector balancing iterations | 20 |
| `fake_quant` | bool | Use fake quantization for testing | False |

## Example

```yaml
meta:
  type: llm

model:
  type: Qwen3
  model_path: /path/to/qwen3-0.6b

calib_dataset:
  dataset:
    type: C4Dataset
    num_samples: 256
  dataloader:
    batch_size: 1

pipeline:
  - type: ptq
    algo_name: QuIP
    algo_config:
      w_bits: 4
      quant_func: rms
      vector_balance_iters: 20

  - type: save
    save_dir: ./outputs/quip
```

## Run Command

```bash
python tools/run.py -c configs/compressor/quip/qwen3_0_6b-w4.yaml
```

## Quantization Functions

### RMS (Recommended)

Uses RMS normalization for scale computation. Better for models with varying weight distributions.

```yaml
quant_func: rms
```

### MinMax

Traditional min-max quantization with explicit zero point.

```yaml
quant_func: minmax
```

## Output Format

QuIP stores quantized weights with dequantization parameters:

```
model.safetensors
├── model.layers.0.mlp.gate_proj.qweight    # Packed int4 weights
├── model.layers.0.mlp.gate_proj.scales     # RMS scale or zero point
└── ...
```

## Deployment

QuIP models can be deployed with vLLM-ascend:

```bash
bash deploy/run_vllm.sh --model-path outputs/quip/qwen3_quip -d 4,5 -t 2 -q quip
```

## Notes

- QuIP does not support group quantization
- RMS mode is generally preferred over MinMax
- Higher `vector_balance_iters` = better quality, slower quantization
- True quantization support is in development (currently uses fake quant)
