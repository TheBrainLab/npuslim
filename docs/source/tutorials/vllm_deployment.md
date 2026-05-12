# vLLM Deployment

Deploy quantized models with vLLM-ascend for high-throughput inference.

## Prerequisites

- vLLM-ascend installed
- Quantized model saved via NPUSlim
- NPUSlim plugin registered (via `pip install -e .`)

## Quick Deploy

```bash
bash deploy/run_vllm.sh --model-path outputs/opt/int8_dynamic/opt_125m-w8a8 -d 4,5 -t 2 -q
```

### Parameters

| Parameter | Description |
|-----------|-------------|
| `--model-path` | Path to quantized model |
| `-d, --devices` | NPU devices to use (e.g., `4,5`) |
| `-t, --tp` | Tensor parallel size |
| `-q, --quantization` | Enable quantization (required for NPUSlim models) |

## Verify Deployment

Start the vLLM server, then test with the chat script:

```bash
python deploy/chat.py
```

Or make an API request:

```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3_int8_dyn",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 100
  }'
```

## Troubleshooting

### Plugin Not Found

Ensure NPUSlim is installed:

```bash
python -c "from npuslim.vllm_plugin import register; register(); print('Plugin registered')"
```

### Quantization Error

Verify the model was quantized with a supported method:

```bash
# Check model config
cat outputs/opt/int8_dynamic/opt_125m-w8a8/config.json | grep quantization_config
```

The `quantization_config` should contain `"quant_method"` matching NPUSlim algorithms.

### OOM During Inference

Reduce tensor parallel size or use smaller models:

```bash
bash deploy/run_vllm.sh --model-path outputs/opt/int8_dynamic/opt_125m-w8a8 -d 4 -t 1 -q
```

## Performance Tuning

- Use multiple NPUs with tensor parallelism (`-t 2, -t 4`)
- Adjust `gpu_memory_utilization` in vLLM config for NPU memory
- Enable KV cache quantization for additional memory savings

## See Also

- [Benchmark Guide](../benchmark/index.md) for evaluation
- [FAQ](../faq/index.md) for common issues
