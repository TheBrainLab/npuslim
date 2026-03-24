from npuslim.registry import TaskRegistry

TaskRegistry.register_lazy("compressor", ".quantize_task", aliases=["QuantizeTask", "CompressorTask"])
