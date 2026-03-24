"""Quantization task using lazy resource manager."""
from __future__ import annotations

from pathlib import Path

from loguru import logger

from npuslim.core.context import AlgorithmContext
from npuslim.core.model_runtime import ModelRuntimeSession
from npuslim.core.step_executor import StepExecutor
from npuslim.registry import TaskRegistry
from npuslim.streaming import StreamSaver
from npuslim.tasks.base_task import BaseTask


@TaskRegistry.register("compressor", aliases=["QuantizeTask", "CompressorTask"])
class QuantizeTask(BaseTask):
    """First concrete task for quantization workflow wiring."""

    def _build_runtime(self, model):
        mode = getattr(self.execution, "mode", "full")
        chunk_size = getattr(self.execution, "chunk_size", 1)
        return ModelRuntimeSession(model=model, mode=mode, chunk_size=chunk_size)

    def _build_stream_saver(self, runtime):
        if not runtime.is_streaming:
            return None
        if not isinstance(self.saver, dict):
            return None

        output_dir = self.saver.get("output_dir") or self.saver.get("save_dir")
        if not output_dir:
            return None

        return StreamSaver(
            output_dir=Path(output_dir),
            shard_size=self.saver.get("shard_size", "5GB"),
            size_threshold=int(self.saver.get("size_threshold", 4 * 1024 * 1024 * 1024)),
        )

    def _chunk_count(self, runtime) -> int:
        if not runtime.is_streaming:
            return 1
        return runtime.get_chunk_count(runtime.chunk_size)

    def _run_algorithm(self, algorithm, context, runtime) -> None:
        steps = algorithm.get_steps()
        if not steps:
            raise ValueError(
                f"Algorithm '{type(algorithm).__name__}' has no @step methods. "
                "Step-based execution is required."
            )

        step_executor = StepExecutor(context, steps)
        chunk_size = runtime.chunk_size
        chunk_count = self._chunk_count(runtime)

        algorithm.on_start(context)
        try:
            for chunk_idx in range(chunk_count):
                algorithm.on_chunk_enter(context)
                chunk_layers = runtime.load_chunk(chunk_idx, chunk_size=chunk_size)
                context.set_current_chunk({"layers": chunk_layers, "index": chunk_idx})
                step_executor.execute()

                context.clear_intermediates()
                context.clear_current_chunk()
                runtime.release_chunk(chunk_idx)
                algorithm.on_chunk_exit(context)
        finally:
            if context.is_streaming and context._stream_saver is not None:
                context._stream_saver.finalize(
                    model_config=getattr(context.model, "config", None),
                    tokenizer=getattr(context.model, "tokenizer", None),
                )
        algorithm.on_finish(context)

    def run(self):
        model = self._resolve_model(self.model_ref)
        processor = getattr(model, "tokenizer", None)
        dataset = self._resolve_dataset(self.data_ref, processor=processor)
        runtime = self._build_runtime(model)
        stream_saver = self._build_stream_saver(runtime)
        context = AlgorithmContext(
            model=model,
            dataloader=dataset,
            runtime=runtime,
            saver=stream_saver,
        )
        algorithm = self._resolve_algorithm()
        if algorithm is None:
            raise ValueError(
                f"Algorithm is not available for task '{self.name}': {getattr(self.algorithm, 'type', None)}"
            )

        algo_name = getattr(self.algorithm, "type", None) if self.algorithm else None
        logger.info(
            f"QuantizeTask wired resources: model={type(model).__name__ if model else None}, "
            f"dataset={type(dataset).__name__ if dataset else None}, algorithm={algo_name}"
        )

        try:
            self._run_algorithm(algorithm=algorithm, context=context, runtime=runtime)
        finally:
            runtime.close()

        if isinstance(self.model_ref, str) and self.model_ref.startswith("@"):
            self.resource_manager.publish_model_state(self.model_ref, model)

        return {
            "model": model,
            "dataset": dataset,
            "algorithm": algorithm or self.algorithm,
            "context": context,
            "saver": self.saver,
        }
