# SparseGPT Quantization

SparseGPT combines structured pruning with quantization for additional compression.

## Overview

- **Sparsification**: Structured pruning using Hessian information
- **Quantization**: 4-bit weight quantization after pruning
- **Two-Phase**: Pruning phase + quantization phase
- **Target Aware**: Can target specific sparsity levels

## Configuration

```yaml
pipeline:
  - type: ptq
    algo_name: SparseGPT
    ignore_layers: []
    algo_config:
      w_bits: 4
      sparsity: 0.5          # Target sparsity (0.0 - 1.0)
      prunen: 0.1            # Pruning ratio per iteration
    calib_config:
      num_samples: 256
```

### Parameters

| Parameter | Type | Description | Default |
|-----------|--------|-------------|----------|
| `w_bits` | int | Weight bit width | 4 |
| `sparsity` | float | Target sparsity ratio (0-1) | 0.5 |
| `prunen` | float | Pruning ratio per iteration | 0.1 |

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

pipeline:
  - type: ptq
    algo_name: SparseGPT
    algo_config:
      w_bits: 4
      sparsity: 0.5

  - type: save
    save_dir: ./outputs/sparsegpt
```

## Run Command

```bash
python tools/run.py -c configs/opt/sparsegpt/opt_125m-s0p5.yaml
```

## Output Format

SparseGPT stores sparse quantized weights:

```
model.safetensors
├── model.layers.0.mlp.gate_proj.qweight    # Sparse packed weights
├── model.layers.0.mlp.gate_proj.scales     # Quantization scales
└── ...
```

## Notes

- Higher `sparsity` = more compression, potential accuracy loss
- Sparse inference requires NPU sparse kernel support
- Calibration dataset quality significantly affects pruning quality
- Use structured sparsity patterns for better hardware efficiency
