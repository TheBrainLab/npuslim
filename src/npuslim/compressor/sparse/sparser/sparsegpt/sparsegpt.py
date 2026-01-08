from loguru import logger
from ..base_sparser import BaseSparser
from .sparsegpt_module import SparseGPTModule
from npuslim.utils.factory import CompressorFactory
from npuslim.compressor.helper.layer_wise_scheduler import LayerWiseScheduler


@CompressorFactory.register()
class SparseGPT(BaseSparser):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.scheduler = None

    def compress(self, dataloader):
        self.slim_model.model.eval()
        self.scheduler = LayerWiseScheduler(self.slim_model, dataloader)

        def sparse_worker(layer_idx, handlers, subset, **kwargs):
            for name in subset.keys():
                if name not in handlers:
                    logger.info(f"Layer {name} is skipped (not in handlers).")
                    continue
                    
                handler = handlers[name]
                handler.fasterprune(
                    layer_name=name,
                    sparsity=kwargs.get("sparsity", 0),
                    prunen=kwargs.get("prunen", 2),
                    prunem=kwargs.get("prunem", 4),
                    percdamp=kwargs.get("percdamp", 0.01),
                    blocksize=kwargs.get("blocksize", 128),
                )

        layers = self.slim_model.get_layers()
        info = self.sparse_info
        self.scheduler.run(
            layers=layers,
            algo_class=SparseGPTModule,
            process_fn=sparse_worker,
            # process_fn 的参数
            sparsity=info.sparsity,
            prunen=info.prunen,
            prunem=info.prunem,
            percdamp=info.algo_specific_params.get("percdamp", 0.01),
            blocksize=info.algo_specific_params.get("blocksize", 128),
        )

    def apply_masks(self): ...
