# src/npuslim/v2/executor.py
"""Pipeline executor for NPUSlim v2."""
from typing import Any, Dict, List, Optional
from loguru import logger

from npuslim.v2.context import AlgorithmContext
from npuslim.v2.algorithm import BaseAlgorithm
from npuslim.v2.step_executor import StepExecutor
from npuslim.v2.hooks import HookDispatcher, HookType


class PipelineExecutor:
    """
    Executes quantization pipeline with streaming support.
    """

    def __init__(self, algorithm: BaseAlgorithm, context: AlgorithmContext):
        self.algorithm = algorithm
        self.context = context
        self.step_executor: Optional[StepExecutor] = None

        self._setup_step_executor()

    def _setup_step_executor(self) -> None:
        """Initialize step executor if algorithm has steps."""
        steps = self.algorithm.get_steps()
        if steps:
            self.step_executor = StepExecutor(self.context, steps)

    def run(self) -> None:
        """Execute the quantization pipeline."""
        # Dispatch on_algorithm_start
        if self.context.hooks:
            self.context.hooks.dispatch(self.context)

        self.algorithm.on_start(self.context)

        try:
            # Process chunks
            for chunk_idx in range(self._get_chunk_count()):
                self._process_chunk(chunk_idx)
        finally:
            # Finalize
            self._finalize()

        self.algorithm.on_finish(self.context)

        # Dispatch on_algorithm_finish
        if self.context.hooks:
            self.context.hooks.dispatch(self.context)

    def _get_chunk_count(self) -> int:
        """Get number of chunks based on execution mode."""
        if self.algorithm.execution_mode.name == "FULL":
            return 1

        # For chunk-wise mode, calculate based on layer count
        layers = self.context.get_layers()
        chunk_size = self.algorithm.chunk_size
        return (len(layers) + chunk_size - 1) // chunk_size

    def _process_chunk(self, chunk_idx: int) -> None:
        """Process a single chunk."""
        # Dispatch on_chunk_enter
        if self.context.hooks:
            self.context.hooks.dispatch(self.context)

        self.algorithm.on_chunk_enter(self.context)

        # Load chunk (if lazy loading)
        chunk_layers = self._load_chunk(chunk_idx)
        self.context.set_current_chunk({"layers": chunk_layers, "index": chunk_idx})

        # Process layers in chunk
        for layer_idx in range(len(chunk_layers)):
            self.context._layer_index = layer_idx

            if self.step_executor:
                self.step_executor.execute()

            # Advance layer
            self.context.advance_layer()

        # Clear intermediates
        self.context.clear_intermediates()

        # Release chunk
        self._release_chunk(chunk_idx)

        self.algorithm.on_chunk_exit(self.context)

        # Dispatch on_chunk_exit
        if self.context.hooks:
            self.context.hooks.dispatch(self.context)

    def _load_chunk(self, chunk_idx: int) -> List[Any]:
        """Load a chunk of layers."""
        # TODO: Implement based on model architecture
        layers = self.context.get_layers()
        chunk_size = self.algorithm.chunk_size
        start = chunk_idx * chunk_size
        end = min(start + chunk_size, len(layers))
        return layers[start:end]

    def _release_chunk(self, chunk_idx: int) -> None:
        """Release a chunk from memory."""
        # TODO: Implement memory cleanup
        pass

    def _finalize(self) -> None:
        """Finalize streaming save."""
        if self.context.is_streaming and self.context._stream_saver:
            self.context._stream_saver.finalize(
                model_config=getattr(self.context.model, "config", None),
                tokenizer=getattr(self.context.model, "tokenizer", None)
            )
