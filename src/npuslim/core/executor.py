# src/npuslim/core/executor.py
"""Pipeline executor for NPUSlim."""
from typing import Any, List, Optional

from npuslim.config.schema import ExecutionMode
from npuslim.core.context import AlgorithmContext
from npuslim.algorithms import BaseAlgorithm
from npuslim.core.step_executor import StepExecutor


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
        if self.algorithm.execution_mode == ExecutionMode.FULL:
            return 1

        # For chunk-wise mode, calculate based on layer count
        total_layers = self.context.get_total_layers()
        chunk_size = self.algorithm.chunk_size
        if total_layers <= 0:
            return 0
        return (total_layers + chunk_size - 1) // chunk_size

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
        chunk_size = self.algorithm.chunk_size
        return self.context.load_chunk(chunk_idx, chunk_size)

    def _release_chunk(self, chunk_idx: int) -> None:
        """Release a chunk from memory."""
        self.context.release_chunk(chunk_idx)

    def _finalize(self) -> None:
        """Finalize streaming save."""
        if self.context.is_streaming and self.context._stream_saver:
            self.context._stream_saver.finalize(
                model_config=getattr(self.context.model, "config", None),
                tokenizer=getattr(self.context.model, "tokenizer", None)
            )
