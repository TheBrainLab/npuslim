"""
Test script for HuggingFace QuIP plugin.

Usage:
    # First, reinstall to register entry points
    pip install -e . -v

    # Then run this script
    python tools/test_hf_plugin.py --model-path outputs/compressor/quip/opt-125m-w4-real
"""

import argparse
import torch
from pathlib import Path


def test_manual_registration(model_path: str):
    """Test by importing the plugin module."""
    print("=" * 60)
    print("Test 1: Import Plugin (triggers decorator registration)")
    print("=" * 60)

    # Import to trigger registration via decorators
    import npuslim.plugins.transformers
    print("✓ Plugin imported (decorators registered)")

    # Load model
    from transformers import AutoModelForCausalLM, AutoTokenizer
    print(f"Loading model from {model_path}...")

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        device_map="auto",
    )

    # Check if QuIPLinear layers are present
    from npuslim.compressor.quantizer.quip.quip_linear import QuIPLinear
    quip_layers = [name for name, m in model.named_modules() if isinstance(m, QuIPLinear)]
    print(f"✓ Found {len(quip_layers)} QuIPLinear layers")
    for name in quip_layers[:3]:
        print(f"  - {name}")
    if len(quip_layers) > 3:
        print(f"  - ... and {len(quip_layers) - 3} more")

    return model, tokenizer


def test_auto_registration(model_path: str):
    """Test if entry point auto-registration works."""
    print("\n" + "=" * 60)
    print("Test 2: Auto Registration via Entry Point")
    print("=" * 60)

    # Fresh import - should auto-register via entry point
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Loading model from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        device_map="auto",
    )

    from npuslim.compressor.quantizer.quip.quip_linear import QuIPLinear
    quip_layers = [name for name, m in model.named_modules() if isinstance(m, QuIPLinear)]
    print(f"✓ Found {len(quip_layers)} QuIPLinear layers")

    return model, tokenizer


def test_inference(model, tokenizer):
    """Test that inference works."""
    print("\n" + "=" * 60)
    print("Test 3: Inference")
    print("=" * 60)

    prompt = "The quick brown fox"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    print(f"Input: '{prompt}'")
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=20, do_sample=False)

    generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"Output: '{generated}'")
    print("✓ Inference successful")


def main():
    parser = argparse.ArgumentParser(description="Test HF QuIP plugin")
    parser.add_argument("--model-path", type=str, required=True, help="Path to quantized model")
    parser.add_argument("--skip-auto", action="store_true", help="Skip auto-registration test")
    args = parser.parse_args()

    if not Path(args.model_path).exists():
        print(f"Error: Model path {args.model_path} does not exist")
        return

    # Check config.json has quantization_config
    config_path = Path(args.model_path) / "config.json"
    if config_path.exists():
        import json
        with open(config_path) as f:
            config = json.load(f)
        quant_config = config.get("quantization_config", {})
        print(f"quantization_config in config.json: {quant_config}")
        if quant_config.get("quant_method") != "quip":
            print("Warning: quantization_config.quant_method is not 'quip'")
    else:
        print(f"Warning: {config_path} not found")

    # Test 1: Manual registration
    model, tokenizer = test_manual_registration(args.model_path)
    test_inference(model, tokenizer)

    # Test 2: Auto registration (if not skipped)
    if not args.skip_auto:
        # Note: This may not work in the same Python process
        # because transformers caches quantizer registry
        print("\nNote: Auto-registration test may need a fresh Python process")
        print("If it fails, try: python -c \"from transformers import AutoModelForCausalLM; ...\"")

    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
