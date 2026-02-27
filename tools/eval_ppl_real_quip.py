#!/usr/bin/env python
"""
Perplexity evaluation script for QuIP real quantization.
Loads QuIPLinear layers and evaluates perplexity on WikiText2.
"""

import argparse
import torch
import torch.nn as nn
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from safetensors.torch import load_file
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from npuslim.compressor.quantizer.quip.quip_linear import QuIPLinear


def replace_linear_with_quip(model, state_dict):
    """
    Replace nn.Linear layers with QuIPLinear and load quantized weights.
    """
    replaced = 0

    for name, module in list(model.named_modules()):
        if isinstance(module, nn.Linear) and 'layers' in name:
            # Check if this layer has QuIPLinear parameters in state_dict
            qweight_key = f"{name}.qweight"
            if qweight_key not in state_dict:
                continue

            # Get parent module
            parts = name.split('.')
            parent = model
            for part in parts[:-1]:
                parent = getattr(parent, part)
            child_name = parts[-1]

            # Get QuIPLinear parameters from state_dict
            qweight = state_dict[f"{name}.qweight"]
            scales = state_dict[f"{name}.scales"]
            scaleWH = state_dict[f"{name}.scaleWH"]
            proj_seed_u = state_dict[f"{name}.proj_seed_u"]
            proj_seed_v = state_dict[f"{name}.proj_seed_v"]

            # Check for zeros (minmax mode)
            zeros_key = f"{name}.zeros"
            has_zero = zeros_key in state_dict
            zeros = state_dict.get(zeros_key, None)

            # Check for bias
            bias_key = f"{name}.bias"
            has_bias = bias_key in state_dict
            bias = state_dict.get(bias_key, None)

            # Get proj_mode from config or default
            proj_mode = 2  # butterfly_nopermute

            # Determine dimensions
            outfeatures = qweight.shape[1]
            bits = 4  # Default to 4-bit
            infeatures = qweight.shape[0] * 32 // bits

            # Create QuIPLinear layer
            quip_linear = QuIPLinear(
                bits=bits,
                infeatures=infeatures,
                outfeatures=outfeatures,
                has_zero=has_zero,
                bias=has_bias,
                proj_mode=proj_mode,
            )

            # Load weights
            quip_linear.qweight.copy_(qweight)
            quip_linear.scales.copy_(scales)
            quip_linear.scaleWH.copy_(scaleWH)
            quip_linear.proj_seed_u.copy_(proj_seed_u)
            quip_linear.proj_seed_v.copy_(proj_seed_v)
            if has_zero and zeros is not None:
                quip_linear.zeros.copy_(zeros)
            if has_bias and bias is not None:
                quip_linear.bias.copy_(bias)

            # Replace the module
            setattr(parent, child_name, quip_linear)
            replaced += 1

    return replaced


@torch.no_grad()
def eval_perplexity(model, testenc, device="cuda"):
    """
    Evaluate perplexity on test set.
    """
    testenc = testenc.input_ids.to(device)

    # Get model's max sequence length
    if hasattr(model.config, 'max_position_embeddings'):
        seqlen = model.config.max_position_embeddings
    else:
        seqlen = 2048

    # Calculate number of samples
    nsamples = testenc.numel() // seqlen
    print(f"Evaluating {nsamples} samples with sequence length {seqlen}")

    # Setup model for evaluation
    use_cache = model.config.use_cache
    model.config.use_cache = False

    # Get model components
    if hasattr(model, 'model'):
        if hasattr(model.model, 'decoder'):
            layers = model.model.decoder.layers
            embed_tokens = model.model.decoder.embed_tokens
            embed_positions = model.model.decoder.embed_positions
            final_layer_norm = getattr(model.model.decoder, 'final_layer_norm', None)
            project_out = getattr(model.model.decoder, 'project_out', None)
        else:
            layers = model.model.layers
            embed_tokens = model.model.embed_tokens
            embed_positions = None
            final_layer_norm = model.model.norm
            project_out = None
    else:
        raise ValueError("Unsupported model architecture")

    # Move embeddings to device
    embed_tokens = embed_tokens.to(device)
    if embed_positions is not None:
        embed_positions = embed_positions.to(device)
    if project_out is not None:
        project_out = project_out.to(device)
    if final_layer_norm is not None:
        final_layer_norm = final_layer_norm.to(device)
    model.lm_head = model.lm_head.to(device)

    # Move first layer to device
    layers[0] = layers[0].to(device)

    # Capture inputs
    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros((nsamples, seqlen, model.config.hidden_size), dtype=dtype, device=device)
    cache = {'i': 0, 'attention_mask': None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            cache['attention_mask'] = kwargs.get('attention_mask', None)
            raise ValueError

    layers[0] = Catcher(layers[0])

    for i in range(nsamples):
        batch = testenc[:, i * seqlen:(i + 1) * seqlen]
        try:
            model(batch)
        except ValueError:
            pass

    layers[0] = layers[0].module

    embed_tokens = embed_tokens.cpu()
    if embed_positions is not None:
        embed_positions = embed_positions.cpu()
    torch.cuda.empty_cache()

    attention_mask = cache['attention_mask']
    outs = torch.zeros_like(inps)

    # Process layers
    for i in tqdm(range(len(layers)), desc="Processing layers"):
        layer = layers[i].to(device)

        for j in range(nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]

        layers[i] = layer.cpu()
        del layer
        torch.cuda.empty_cache()
        inps, outs = outs, inps

    # Compute perplexity
    if final_layer_norm is not None:
        final_layer_norm = final_layer_norm.to(device)
    if project_out is not None:
        project_out = project_out.to(device)

    lm_head = model.lm_head.to(device)

    nlls = []
    for i in tqdm(range(nsamples), desc="Computing perplexity"):
        hidden_states = inps[i].unsqueeze(0)

        if final_layer_norm is not None:
            hidden_states = final_layer_norm(hidden_states)
        if project_out is not None:
            hidden_states = project_out(hidden_states)

        lm_logits = lm_head(hidden_states)
        shift_logits = lm_logits[:, :-1, :].contiguous()
        shift_labels = testenc[:, i * seqlen:(i + 1) * seqlen][:, 1:]

        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        neg_log_likelihood = loss.float() * seqlen
        nlls.append(neg_log_likelihood)

    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * seqlen))

    model.config.use_cache = use_cache
    return ppl.item()


def load_wikitext2(tokenizer, seqlen=2048):
    """Load WikiText2 test dataset."""
    from datasets import load_dataset

    testdata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')
    testenc = tokenizer('\n\n'.join(testdata['text']), return_tensors='pt')

    return testenc


def main():
    parser = argparse.ArgumentParser(description='Evaluate QuIP real quantization perplexity')
    parser.add_argument('model_path', type=str, help='Path to model checkpoint')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    parser.add_argument('--seqlen', type=int, default=2048, help='Sequence length')
    args = parser.parse_args()

    print(f"Loading model from {args.model_path}...")

    # Load config
    config = AutoConfig.from_pretrained(args.model_path, trust_remote_code=True)

    # Remove quantization_config to avoid conflicts
    if hasattr(config, 'quantization_config'):
        delattr(config, 'quantization_config')

    # Load base model with standard architecture
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        config=config,
        torch_dtype='auto',
        device_map='cpu',
        trust_remote_code=True,
    )

    # Load state dict from safetensors
    safetensors_path = os.path.join(args.model_path, "model.safetensors")
    if os.path.exists(safetensors_path):
        print("Loading quantized weights from safetensors...")
        state_dict = load_file(safetensors_path)
    else:
        print("Loading quantized weights from pytorch_model.bin...")
        state_dict = torch.load(os.path.join(args.model_path, "pytorch_model.bin"))

    # Replace Linear with QuIPLinear
    replaced = replace_linear_with_quip(model, state_dict)
    print(f"Replaced {replaced} layers with QuIPLinear")

    model.eval()

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

    # Load WikiText2
    print("Loading WikiText2 test set...")
    testenc = load_wikitext2(tokenizer, seqlen=args.seqlen)

    # Evaluate
    print("Evaluating perplexity...")
    ppl = eval_perplexity(model, testenc, device=args.device)

    print(f"\n{'='*50}")
    print(f"WikiText2 Perplexity: {ppl:.2f}")
    print(f"{'='*50}")

    return ppl


if __name__ == '__main__':
    main()
