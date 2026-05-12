# GPTQ Quantization

GPTQ (Group-wise Post-training Quantization) is an activation-aware weight quantization method that achieves high accuracy with 4-bit weights.

## Overview

- **Weight Quantization**: 4-bit, group-wise
- **Activation-Aware**: Uses Hessian information for optimal rounding
- **Two-Phase**: Calibration phase + conversion phase
- **True Quantization**: Stores packed int4 weights with dequantization parameters

## Configuration

```yaml
pipeline:
  - type: ptq
    algo_name: GPTQ
    ignore_layers: []
    algo_config:
      w_bits: 4
      group_size: 128
      actorder: True
      damp_percent: 0.01
    calib_config:
      num_samples: 256
```

### Parameters

| Parameter | Type | Description | Default |
|-----------|--------|-------------|----------|
| `w_bits` | int | Weight bit width (usually 4) | 4 |
| `group_size` | int | Group size for quantization | 128 |
| `actorder` | bool | Sort activation by magnitude | True |
| `damp_percent` | float | Damping factor for Hessian | 0.01 |

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
    algo_name: GPTQ
    algo_config:
      w_bits: 4
      group_size: 128
      actorder: True

  - type: eval

  - type: save
    save_dir: ./outputs/gptq
```

## Run Command

```bash
python tools/run.py -c configs/opt/gptq/opt_125m-w4a16.yaml
```

## Output Format

GPTQ stores quantized weights in packed format:

```
model.safetensors
├── model.layers.0.mlp.gate_proj.qweight    # Packed int4 weights
├── model.layers.0.mlp.gate_proj.qzeros    # Packed zero points
├── model.layers.0.mlp.gate_proj.scales     # Float16 scales
├── model.layers.0.mlp.gate_proj.g_idx      # Group indices
└── ...
```

## Deployment

GPTQ models can be deployed with vLLM-ascend:

```bash
bash deploy/run_vllm.sh --model-path outputs/gptq/qwen3_gptq -d 4,5 -t 2 -q gptq
```

## Notes

- Smaller `group_size` = better accuracy, slower inference
- Larger `group_size` = faster inference, lower accuracy
- Use `actorder=True` for better quality
- Hessian computation requires calibration data
