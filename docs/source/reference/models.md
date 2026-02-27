# Models

## BaseLLMModel

```{eval-rst}
.. autoclass:: npuslim.model.base_model.BaseLLMModel
   :members:
   :undoc-members:
   :show-inheritance:
```

## Usage Example

```python
from npuslim import ModelFactory
from npuslim.utils.config_parser import ModelConfig

# Create config
config = ModelConfig(
    type="Qwen3",
    model_path="Qwen/Qwen3-0.6B",
    model_hub="hf"
)

# Create model
model = ModelFactory.create(config=config)
model.prepare()

# Access tokenizer
tokenizer = model.tokenizer
```

## Model Adapters

### Qwen3

Qwen3 model adapter with Ascend NPU support.

**File**: `src/npuslim/model/qwen3/qwen3_model.py`

```{eval-rst}
.. autoclass:: npuslim.model.qwen3.qwen3_model.Qwen3SlimModel
   :members:
   :undoc-members:
```

### OPT

OPT model adapter.

**File**: `src/npuslim/model/opt/opt_model.py`

```{eval-rst}
.. autoclass:: npuslim.model.opt.opt_model.OPTSlimModel
   :members:
   :undoc-members:
```

## Adding Custom Models

See [Adding Custom Model Support](../tutorials/custom_model.md) for a complete guide.
