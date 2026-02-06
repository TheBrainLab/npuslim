import torch
from typing import Dict, Any, Optional
from tqdm import tqdm
from loguru import logger
from npuslim.utils.utils import find_layers
from npuslim.utils.backend import bh


def collect_initial_inputs(model, target_layer, dataloader):
    first_batch = next(iter(dataloader))
    nsamples = len(dataloader)
    config = model.model.config
    hidden_size = getattr(config, "hidden_size", getattr(config, "d_model", None))
    seqlen = first_batch["input_ids"].shape[1]
    dtype = next(model.model.parameters()).dtype
    inps = torch.zeros((nsamples, seqlen, hidden_size), dtype=dtype, device=bh.device)

    layer_kwargs = {}
    cache = {"i": 0}

    pre_transformer_modules_dict = model.get_pre_transformer_modules()
    for _, module in pre_transformer_modules_dict.items():
        module.to(bh.device)

    def hook_fn(module, args, kwargs):
        curr_inp = args[0]
        batch_size = curr_inp.shape[0]
        inps[cache["i"] : cache["i"] + batch_size] = curr_inp.detach()
        cache["i"] += batch_size

        if not layer_kwargs:
            for k, v in kwargs.items():
                if isinstance(v, torch.Tensor):
                    layer_kwargs[k] = v.detach().to(bh.device)
                else:
                    layer_kwargs[k] = v

        raise ValueError("Data captured")

    handle = target_layer.register_forward_pre_hook(hook_fn, with_kwargs=True)

    logger.info(f"Capturing activations...")
    with torch.no_grad():
        for batch in tqdm(dataloader):
            try:
                batch = {
                    k: v.to(bh.device) if torch.is_tensor(v) else v
                    for k, v in batch.items()
                }
                model.forward(**batch)
            except ValueError:
                continue

    handle.remove()
    for _, module in pre_transformer_modules_dict.items():
        module.cpu()
    bh.empty_cache()

    return inps, layer_kwargs


def execute_optimization_worker(
    layer_idx: int, 
    handlers: Dict[str, Any], 
    subset: Dict[str, Any], 
    method_name: str = "fasterquant", 
    results_dict: Optional[Dict] = None, # 显式接收结果字典
    **kwargs
):
    total_sub_layers = len(subset)
    for i, (name, layer_module) in enumerate(subset.items()):
        if name not in handlers:
            logger.info(f"Layer {name} is skipped (not in handlers).")
            continue

        logger.info(
            f"-> [Layer {layer_idx}] Optimizing module ({i+1}/{total_sub_layers}): {name}"
        )
        
        handler = handlers[name]
        if not hasattr(handler, method_name):
            logger.warning(f"Handler for {name} does not have method '{method_name}'. Skipping.")
            continue
        
        process_func = getattr(handler, method_name)
        result = process_func(layer_name=name, **kwargs)
        if result is not None and results_dict is not None:
            results_dict[name] = result


class LayerWiseScheduler:
    def __init__(self, model, dataloader):
        self.model = model
        self.dataloader = dataloader
        self.nsamples = len(dataloader)

    def _prepare_kwargs(self, layer_kwargs):
        return {
            k: v.to(bh.device) if isinstance(v, torch.Tensor) else v
            for k, v in layer_kwargs.items()
        }

    def get_layer_path_mapping(self, layers_list):
        id_to_name = {id(m): name for name, m in self.model.model.named_modules()}
        mapping = {}
        for layer in layers_list:
            obj_id = id(layer)
            if obj_id in id_to_name:
                mapping[obj_id] = id_to_name[obj_id]
            else:
                mapping[obj_id] = "unknown_path"
        return mapping

    def run(
        self,
        layers,
        algo_class,
        process_fn,
        algo_config=None,
        ignore_layers=None,
        **kwargs, 
    ):
        inps, layer_kwargs = collect_initial_inputs(
            self.model, layers[0], self.dataloader
        )
        outs = torch.zeros_like(inps)
        layer_kwargs = self._prepare_kwargs(layer_kwargs)

        path_mapping = self.get_layer_path_mapping(layers)
        ignore_layers = ignore_layers or []
        for i, layer in enumerate(layers):
            logger.info(f"Start optimizing Layer {i+1}/{len(layers)}...")

            layer.to(bh.device)
            relative_subset = find_layers(layer)

            abs_layer_prefix = path_mapping.get(id(layer))
            handlers = {}
            full_subset = {}

            for n, m in relative_subset.items():
                full_path = f"{abs_layer_prefix}.{n}"
                full_subset[full_path] = m

                if full_path in ignore_layers:
                    logger.info(f"Layer {full_path} is skipped (in ignore list).")
                    continue

                handler = algo_class(m, config=algo_config)
                handlers[full_path] = handler

            if not handlers:
                logger.warning(
                    f"No target layers found in Layer {i}, skipping statistics."
                )
            else:
                self._collect_statistics(layer, handlers, inps, outs, layer_kwargs)
                process_fn(i, handlers, full_subset, **kwargs)

            if i < len(layers) - 1:
                for j in tqdm(
                    range(self.nsamples),
                    desc=f"Calibrating Layer {i+1}/{len(layers)}",
                ):
                    outs[j] = layer(inps[j].unsqueeze(0), **layer_kwargs)[0].detach()
                inps, outs = outs, inps
            else:
                logger.info("Last layer reached, skipping final forward pass.")

            self._cleanup(layer, handlers)
            logger.info(f"Layer {i+1}/{len(layers)} optimized.")

    def _collect_statistics(self, layer, handlers, inps, outs, layer_kwargs):
        id_to_handler_key = {id(h.layer): k for k, h in handlers.items()}

        def get_hook(name):
            return lambda m, inp, out: handlers[name].add_batch(inp[0].data, out.data)

        handles = []
        for n, m in layer.named_modules():
            m_id = id(m)
            if m_id in id_to_handler_key:
                full_name = id_to_handler_key[m_id]
                handles.append(m.register_forward_hook(get_hook(full_name)))

        for j in range(self.nsamples):
            layer(inps[j].unsqueeze(0), **layer_kwargs)

        for h in handles:
            h.remove()
        
        for h in handlers.values():
            h.preproc()

    def _cleanup(self, layer, handlers):
        layer.cpu()
        for h in handlers.values():
            h.free()
        bh.empty_cache()
