# FAQ

## General

### How do I handle OOM during quantization?

**Answer**: Reduce the calibration dataset size or batch size:

```yaml
calib_dataset:
  dataset:
    num_samples: 128  # Reduce from 256
  dataloader:
    batch_size: 1
```

### How do I add a custom quantization algorithm?

**Answer**: Implement the `BaseCompressorAlgo` interface:

```python
from npuslim.compressor.quantizer.base_algo import BaseCompressorAlgo
from npuslim.utils.factory import CompressorFactory

@CompressorFactory.register(name="my_algo")
class MyQuantizer(BaseCompressorAlgo):
    def __init__(self, model, config, dataloader, ignore_layers, **kwargs):
        super().__init__(model, config, dataloader, ignore_layers)

    def prepare(self):
        # Preparation phase
        pass

    def calibrate(self):
        # Calibration phase
        pass

    def convert(self):
        # Conversion phase
        pass
```

Then use in config:

```yaml
pipeline:
  - type: ptq
    algo_name: my_algo
```

### vLLM deployment fails with quantization error

**Answer**: Ensure the quantization method is registered:

1. Verify NPUSlim installation:

```bash
python -c "from npuslim.vllm_plugin import register; register(); print('OK')"
```

2. Check model config:

```bash
cat outputs/opt/int8_dynamic/opt_125m-w8a8/config.json | grep quantization_config
```

3. Ensure `-q` flag is passed to vLLM:

```bash
bash deploy/run_vllm.sh --model-path outputs/opt/int8_dynamic/opt_125m-w8a8 -q
```

### How do I evaluate quantization accuracy?

**Answer**: Use perplexity evaluation:

```bash
# WikiText2
python tools/eval_ppl_quip_style.py --model outputs/opt/int8_dynamic/opt_125m-w8a8

# LM-Eval
bash tools/eval/run_lmeval.sh outputs/opt/int8_dynamic/opt_125m-w8a8 --tasks wikitext
```

Compare with the original FP16 model to measure degradation.

### Which model architectures are supported?

**Answer**: Currently supported:

- Qwen3
- OPT

More models can be added. See [Adding Custom Model Support](../tutorials/custom_model.md).

## Troubleshooting

### CANN not found

**Error**: `ASCEND_HOME_PATH not set`

**Solution**: Source CANN environment:

```bash
source /path/to/cann/set_env.sh
```

### Import error for transformers

**Error**: `ImportError: No module named 'transformers'`

**Solution**: Install dependencies:

```bash
pip install transformers<5.0.0
```

### CUDA device not available

**Error**: `RuntimeError: CUDA device not available`

**Solution**: Set device_map to NPU:

```yaml
model:
  model_kwargs:
    device_map: auto
```

Or use NPU-specific backend in your code.
