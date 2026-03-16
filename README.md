# NPUSlim

NPU-oriented model compression & quantization framework for large language models.

## Features

- Post-training quantization: INT8Dynamic, GPTQ, QuIP, SparseGPT
- Model support: Qwen3, OPT
- vLLM-ascend deployment integration
- Performance evaluation with lm-eval and evalscope

## Installation

```bash
pip install -e .
```

Requires CANN environment with `ASCEND_HOME_PATH` set.

## Quick Start

### Quantization

```bash
export HF_ENDPOINT="https://hf-mirror.com"  # if need
python tools/run.py -c configs/compressor/int8_dyn/qwen3_0_6b.yaml
```

### Deployment (vLLM)

```bash
# Deploy vLLM server
bash tools/serve/deploy_vllm.sh outputs/qwen3_int8_dyn -d 0,1 -t 2 --wait

# Or run stress test pipeline (deploy + benchmark + cleanup)
bash tools/eval/run_stress_test.sh outputs/qwen3_int8_dyn -d 0,1 -t 2
```

### Evaluation

```bash
# LM-Eval (wikitext, ceval, etc.)
bash tools/eval/run_lmeval.sh outputs/qwen_int8_dyn --tasks wikitext -d 0,1 -t 2

# Stress test with evalscope (requires running vLLM server first)
bash tools/eval/run_stress_test.sh outputs/qwen_int8_dyn --parallel "1 16 32"
```

## Tool Scripts

| Script | Description |
|-------|-------------|
| `tools/serve/deploy_vllm.sh` | Deploy vLLM inference server |
| `tools/eval/run_lmeval.sh` | Run lm-evaluation-harness benchmarks |
| `tools/eval/run_stress_test.sh` | Full pipeline: deploy → stress test → cleanup |

### Common Options

All scripts support:
- `-d, --devices` - Device IDs (e.g., `0,1` or `4,5`)
- `-t, --tp` - Tensor parallel size
- `--gpu-memory` - GPU memory utilization (default: 0.8)
- `--max-model-len` - Max model length (default: 4096)
- `-q, --quantization` - Quantization method (auto-detected on NPU)

Use `--help` to see all options for each script.

## Configuration

Edit config files in `configs/compressor/` to customize model path, quantization parameters, and pipeline tasks.

```yaml
model:
  model_path: your/model/path

pipeline:
  - type: ptq
    algo_name: INT8Dynamic
```

## Architecture

- `src/npuslim/slim_engine.py` - Orchestrator managing resources and task pipeline
- `src/npuslim/utils/factory.py` - Factory pattern for models, datasets, tasks, compressors
- `src/npuslim/compressor/quantizer/` - Quantization algorithms (INT8Dynamic, GPTQ, QuIP, SparseGPT)
- `src/npuslim/vllm_plugin/` - vLLM-ascend integration
- `tools/utils/common.sh` - Shared bash utilities (logging, device detection)

## License

Apache-2.0
