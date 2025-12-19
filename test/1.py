from importlib.metadata import entry_points
eps = entry_points().select(group='vllm.general_quantization', name='npuslim')
print(list(eps))