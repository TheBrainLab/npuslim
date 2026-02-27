# Observers

## BaseObserver

```{eval-rst}
.. autoclass:: npuslim.compressor.observers.base_observer.BaseObserver
   :members:
   :undoc-members:
   :show-inheritance:
```

## Observer Classes

### AbsMaxTokenWiseActObserver

Absolute maximum observer for activation statistics (per-token).

```{eval-rst}
.. autoclass:: npuslim.compressor.observers.abs_max_activation.AbsMaxTokenWiseActObserver
   :members:
   :undoc-members:
```

### AbsmaxPerchannelObserver

Absolute maximum observer for activation statistics (per-channel).

```{eval-rst}
.. autoclass:: npuslim.compressor.observers.abs_max_activation.AbsmaxPerchannelObserver
   :members:
   :undoc-members:
```

### AbsMaxChannelWiseWeightObserver

Absolute maximum observer for weight statistics (per-channel).

```{eval-rst}
.. autoclass:: npuslim.compressor.observers.abs_max_weight.AbsMaxChannelWiseWeightObserver
   :members:
   :undoc-members:
```

### PTQObserver

Post-training quantization observer.

```{eval-rst}
.. autoclass:: npuslim.compressor.observers.ptq_observer.PTQObserver
   :members:
   :undoc-members:
```

## Usage

```python
from npuslim.compressor.observers import AbsMaxTokenWiseActObserver

# Create observer
observer = AbsMaxTokenWiseActObserver(shape=(out_features,), dtype=torch.float32)

# Collect statistics
observer.update(activation_tensor)

# Get quantization parameters
scale, zero = observer.get_scale_zero()
```

## Observer Selection

Models specify observer layer classes via `observer_layer_classes`:

```python
class MyModel(BaseLLMModel):
    observer_layer_classes = [torch.nn.Linear, torch.nn.Conv2d]
```
