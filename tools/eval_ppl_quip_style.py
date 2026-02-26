#!/usr/bin/env python
"""
Perplexity evaluation script matching official QuIP implementation.
This computes perplexity on WikiText2 test set using the same method as QuIP.

Reference: https://github.com/Cornell-RelaxML/QuIP/blob/master/opt.py
"""

import argparse
import torch
import torch.nn as nn
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


@torch.no_grad()
def eval_perplexity(model, testenc, device="cuda"):
    """
    Evaluate perplexity on test set, matching official QuIP implementation.

    Args:
        model: The model to evaluate
        testenc: Tokenized test dataset (input_ids)
        device: Device to run evaluation on

    Returns:
        perplexity (float)
    """
    testenc = testenc.input_ids.to(device)

    # Get model's max sequence length
    if hasattr(model.config, 'max_position_embeddings'):
        seqlen = model.config.max_position_embeddings
    else:
        seqlen = 2048

    # Calculate number of samples (entire test set divided by seqlen)
    nsamples = testenc.numel() // seqlen
    print(f"Evaluating {nsamples} samples with sequence length {seqlen}")

    # Setup model for evaluation
    use_cache = model.config.use_cache
    model.config.use_cache = False

    # Get model components
    if hasattr(model, 'model'):
        if hasattr(model.model, 'decoder'):
            # OPT model
            layers = model.model.decoder.layers
            embed_tokens = model.model.decoder.embed_tokens
            embed_positions = model.model.decoder.embed_positions
            final_layer_norm = getattr(model.model.decoder, 'final_layer_norm', None)
            project_out = getattr(model.model.decoder, 'project_out', None)
        else:
            # LLaMA-style model
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

    # Move first layer to device for capturing inputs
    layers[0] = layers[0].to(device)

    # Capture inputs to each layer
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

    # Run forward pass to capture inputs
    for i in range(nsamples):
        batch = testenc[:, i * seqlen:(i + 1) * seqlen]
        try:
            model(batch)
        except ValueError:
            pass

    layers[0] = layers[0].module

    # Move embeddings back to CPU
    embed_tokens = embed_tokens.cpu()
    if embed_positions is not None:
        embed_positions = embed_positions.cpu()
    torch.cuda.empty_cache()

    # Store attention mask
    attention_mask = cache['attention_mask']
    outs = torch.zeros_like(inps)

    # Process through each layer
    for i in tqdm(range(len(layers)), desc="Processing layers"):
        layer = layers[i].to(device)

        for j in range(nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]

        layers[i] = layer.cpu()
        del layer
        torch.cuda.empty_cache()
        inps, outs = outs, inps

    # Compute perplexity using final hidden states
    if final_layer_norm is not None:
        final_layer_norm = final_layer_norm.to(device)
    if project_out is not None:
        project_out = project_out.to(device)

    # Ensure lm_head is on device
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

    # Compute final perplexity
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
    parser = argparse.ArgumentParser(description='Evaluate perplexity (QuIP style)')
    parser.add_argument('model_path', type=str, help='Path to model checkpoint')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    parser.add_argument('--seqlen', type=int, default=2048, help='Sequence length')
    args = parser.parse_args()

    # Load model and tokenizer
    print(f"Loading model from {args.model_path}...")

    # Load config and remove quantization_config if present (we have fake quantized weights)
    from transformers import AutoConfig
    config = AutoConfig.from_pretrained(args.model_path, trust_remote_code=True)
    if hasattr(config, 'quantization_config'):
        delattr(config, 'quantization_config')
        print("Removed quantization_config from model (using fake quantization)")

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        config=config,
        torch_dtype='auto',
        device_map='cpu',
        trust_remote_code=True,
    )
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

    # Load WikiText2 test set
    print("Loading WikiText2 test set...")
    testenc = load_wikitext2(tokenizer, seqlen=args.seqlen)

    # Evaluate perplexity
    print("Evaluating perplexity...")
    ppl = eval_perplexity(model, testenc, device=args.device)

    print(f"\n{'='*50}")
    print(f"WikiText2 Perplexity: {ppl:.2f}")
    print(f"{'='*50}")

    return ppl


if __name__ == '__main__':
    main()
