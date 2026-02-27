# NPUSlim

NPU-oriented model compression & quantization framework for large language models.

## Features

- Post-training quantization: INT8Dynamic, GPTQ, QuIP, SparseGPT
- Model support: Qwen3, OPT
- vLLM-ascend deployment integration
- Performance evaluation with lm-eval and evalscope

## Installation

```bash
pip install -e . -v
```

Requires CANN environment with `ASCEND_HOME_PATH` set.

## Quick Start

### Quantization

```bash
python tools/run.py -c configs/compressor/int8_dyn/qwen3_0_6b.yaml
```

### Deployment

```bash
bash deploy/run_vllm.sh --model-path outputs/qwen3_int8_dyn -d 4,5 -t 2 -q
```

### Evaluation

```bash
# LM-Eval
bash tools/eval/run_lmeval.sh outputs/qwen_int8_dyn --tasks wikitext

# Stress test
bash tools/eval/evalscope_perf.sh outputs/qwen_int8_dyn
```

## Configuration

Edit config files in `configs/compressor/` to customize model path, quantization parameters, and pipeline tasks.

```yaml
model:
  model_path: your/model/path

pipeline:
  - type: ptq
    algo_name: INT8Dynamic
```

## License

Apache-2.0
