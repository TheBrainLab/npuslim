# tests/algorithms/test_base_algo.py
import torch
from npuslim.algorithms.base_algo import BaseAlgorithm
from npuslim.tasks.compressor.context import ChunkContext, LayerInfo


class DummyAlgo(BaseAlgorithm):
    """Simple test algorithm that doubles tensors."""

    def process_chunk(self, chunk: ChunkContext) -> ChunkContext:
        for layer in chunk.layers:
            for name, tensor in list(layer.tensors.items()):
                layer.tensors[name] = tensor * 2
        return chunk


def test_base_algorithm_process_chunk():
    algo = DummyAlgo()

    chunk = ChunkContext(
        chunk_index=0,
        layers=[
            LayerInfo(name="layer.0", index=0, tensors={"w": torch.tensor(1.0)}),
        ],
    )

    result = algo.process_chunk(chunk)
    assert result.layers[0].tensors["w"].item() == 2.0


def test_base_algorithm_lifecycle():
    algo = DummyAlgo()

    algo.on_start()
    chunk = ChunkContext(
        chunk_index=0,
        layers=[LayerInfo(name="l", index=0, tensors={"x": torch.tensor(1.0)})],
    )
    result = algo.process_chunk(chunk)
    algo.on_finish()

    assert result.layers[0].tensors["x"].item() == 2.0
