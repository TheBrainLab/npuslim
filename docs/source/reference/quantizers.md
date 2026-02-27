# Quantizers

## BaseCompressorAlgo

```{eval-rst}
.. autoclass:: npuslim.compressor.quantizer.base_algo.BaseCompressorAlgo
   :members:
   :undoc-members:
   :show-inheritance:
```

## Quantization Algorithms

### INT8Dynamic

Per-channel weight, per-token activation quantization.

**File**: `src/npuslim/compressor/quantizer/int8_dyn/int8_dyn.py`

```{eval-rst}
.. autoclass:: npuslim.compressor.quantizer.int8_dyn.int8_dyn.INT8Dynamic
   :members:
   :undoc-members:
```

### GPTQ

Group-wise post-training quantization with Hessian information.

**File**: `src/npuslim/compressor/quantizer/gptq/gptq.py`

```{eval-rst}
.. autoclass:: npuslim.compressor.quantizer.gptq.gptq.GPTQ
   :members:
   :undoc-members:
```

### QuIP

Quaternion-inspired quantization with LDLQ decomposition.

**File**: `src/npuslim/compressor/quantizer/quip/quip.py`

```{eval-rst}
.. autoclass:: npuslim.compressor.quantizer.quip.quip.QuIP
   :members:
   :undoc-members:
```

### SparseGPT

Structured pruning with quantization.

**File**: `src/npuslim/compressor/quantizer/sparsegpt/sparsegpt.py`

```{eval-rst}
.. autoclass:: npuslim.compressor.quantizer.sparsegpt.sparsegpt.SparseGPT
   :members:
   :undoc-members:
```

## Usage

```python
from npuslim import CompressorFactory
from npuslim.compressor.quantizer.base_algo import BaseCompressorAlgo

# Create quantizer
quantizer = CompressorFactory.create(
    algo_name="INT8Dynamic",
    model=model,
    config={"w_bits": 8},
    dataloader=dataloader,
    ignore_layers=["lm_head"]
)

# Run quantization
quantizer.prepare()
quantizer.calibrate()
quantizer.convert()
```

## Common Parameters

| Parameter | Type | Description |
|-----------|--------|-------------|
| `model` | BaseLLMModel | Target model |
| `config` | dict | Algorithm-specific config |
| `dataloader` | DataLoader | Calibration data |
| `ignore_layers` | list | Layers to skip |
