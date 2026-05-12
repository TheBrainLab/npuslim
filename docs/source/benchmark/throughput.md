# Throughput Benchmarks

[TO BE ADDED]

## Overview

vLLM-ascend throughput measurements for quantized models.

## Test Setup

- **Hardware**: [Specify NPU model]
- **Model**: [Model name and size]
- **Batch Size**: [Tested batch sizes]
- **Sequence Length**: [Tested sequence lengths]

## Results

| Quantization | Throughput (tokens/s) | Latency (ms/token) | Memory (GB) |
|-------------|------------------------|---------------------|---------------|
| FP16 | - | - | - |
| INT8Dynamic | - | - | - |
| GPTQ-4bit | - | - | - |
| QuIP-4bit | - | - | - |

## Evaluation Method

```bash
# Deploy vLLM server
bash deploy/run_vllm.sh --model-path outputs/opt/int8_dynamic/opt_125m-w8a8 -d 4,5 -t 2 -q

# Run stress test
bash tools/eval/evalscope_perf.sh outputs/opt/int8_dynamic/opt_125m-w8a8 --parallel "1 16 32 64"
```

## Notes

[Analysis of throughput characteristics]
