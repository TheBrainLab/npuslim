from loguru import logger
from functools import partial
from ..base_sparser import BaseSparser
from .sparsegpt_module import SparseGPTModule
from npuslim.utils.factory import CompressorFactory
from npuslim.compressor.helper.layer_wise_scheduler import LayerWiseScheduler


@CompressorFactory.register()
class SparseGPT(BaseSparser):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.scheduler = None

    def prepare(self):
        self.slim_model.model.eval()
        self.scheduler = LayerWiseScheduler(self.slim_model, self.dataloader)

    def compress(self):
        def sparse_worker(layer_idx, handlers, subset, **kwargs):
            total_sub_layers = len(subset)
            for i, name in enumerate(subset.keys()):
                if name not in handlers:
                    logger.info(f"Layer {name} is skipped (not in handlers).")
                    continue
                logger.info(
                    f"-> [Layer {layer_idx}] Optimizing module ({i+1}/{total_sub_layers}): {name}"
                )
                handler = handlers[name]
                handler.process()

        layers = self.slim_model.get_layers()
        info = self.sparse_info
        prune_algo = partial(
            SparseGPTModule,
            sparsity=info.sparsity,
            prunen=info.prunen,
            prunem=info.prunem,
            percdamp=info.algo_specific_params.get("percdamp", 0.01),
            blocksize=info.algo_specific_params.get("blocksize", 128),
        )
        self.scheduler.run(
            layers=layers,
            algo_class=prune_algo,
            process_fn=sparse_worker,
            ignore_layers=self.ignore_layers,
        )

    def apply_masks(self): ...
