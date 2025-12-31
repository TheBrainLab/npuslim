import torch
from tqdm import tqdm
from loguru import logger
from npuslim.utils.utils import find_layers


def collect_initial_inputs(model, target_layer, dataloader, device="npu"):
    if device == "npu" and not torch.npu.is_available():
        logger.warning("NPU not detected. Falling back to CPU.")
        device = "cpu"

    first_batch = next(iter(dataloader))
    nsamples = len(dataloader)
    config = model.model.config
    hidden_size = getattr(config, "hidden_size", getattr(config, "d_model", None))
    seqlen = first_batch["input_ids"].shape[1]
    dtype = next(model.model.parameters()).dtype
    inps = torch.zeros((nsamples, seqlen, hidden_size), dtype=dtype, device=device)

    layer_kwargs = {}
    cache = {"i": 0}

    pre_transformer_modules_dict = model.get_pre_transformer_modules()
    for _, module in pre_transformer_modules_dict.items():
        module.to(device)

    def hook_fn(module, args, kwargs):
        curr_inp = args[0]
        batch_size = curr_inp.shape[0]
        inps[cache["i"] : cache["i"] + batch_size] = curr_inp.detach()
        cache["i"] += batch_size

        if not layer_kwargs:
            for k, v in kwargs.items():
                if isinstance(v, torch.Tensor):
                    layer_kwargs[k] = v.detach().to(device)
                else:
                    layer_kwargs[k] = v

        raise ValueError("Data captured")

    handle = target_layer.register_forward_pre_hook(hook_fn, with_kwargs=True)

    logger.info(f"Capturing activations...")
    with torch.no_grad():
        for batch in tqdm(dataloader):
            try:
                batch = {
                    k: v.to(device) if torch.is_tensor(v) else v
                    for k, v in batch.items()
                }
                model.forward(**batch)
            except ValueError:
                continue

    handle.remove()
    for _, module in pre_transformer_modules_dict.items():
        module.cpu()
    torch.npu.empty_cache()

    return inps, layer_kwargs


class LayerWiseScheduler:
    def __init__(self, model, dataloader, device="npu"):
        self.model = model
        self.dataloader = dataloader
        self.device = device
        self.nsamples = len(dataloader)

    def _prepare_kwargs(self, layer_kwargs):
        return {
            k: v.to(self.device) if isinstance(v, torch.Tensor) else v
            for k, v in layer_kwargs.items()
        }

    def run(self, layers, algo_class, process_fn, **kwargs):
        inps, layer_kwargs = collect_initial_inputs(
            self.model, layers[0], self.dataloader, self.device
        )
        outs = torch.zeros_like(inps)
        layer_kwargs = self._prepare_kwargs(layer_kwargs)

        for i, layer in enumerate(layers):
            logger.info(f"Start optimizing Layer {i+1}/{len(layers)}...")
            
            layer.to(self.device)
            subset = find_layers(layer)

            ignore_list = kwargs.get("ignore_layers", [])
            handlers = {}
            for n, m in subset.items():
                if n not in ignore_list:
                    handler = algo_class(m)
                    handler.layer_name = n
                    handlers[n] = handler

            if not handlers:
                logger.warning(
                    f"No target layers found in Layer {i}, skipping statistics."
                )
            else:
                self._collect_statistics(layer, handlers, inps, outs, layer_kwargs)
                process_fn(i, handlers, subset, **kwargs)

            for j in range(self.nsamples):
                outs[j] = layer(inps[j].unsqueeze(0), **layer_kwargs)[0].detach()

            self._cleanup(layer, handlers)
            inps, outs = outs, inps
            logger.info(f"Layer {i+1}/{len(layers)} optimized.")

    def _collect_statistics(self, layer, handlers, inps, outs, layer_kwargs):
        def get_hook(name):
            return lambda m, inp, out: handlers[name].add_batch(inp[0].data, out.data)

        handles = [
            m.register_forward_hook(get_hook(n))
            for n, m in layer.named_modules()
            if n in handlers
        ]

        for j in range(self.nsamples):
            layer(inps[j].unsqueeze(0), **layer_kwargs)

        for h in handles:
            h.remove()

    def _cleanup(self, layer, handlers):
        layer.cpu()
        for h in handlers.values():
            h.free()
        torch.npu.empty_cache()
