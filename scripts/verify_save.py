#!/usr/bin/env python
"""Verification script to inspect saved quantized models.

Usage:
  python verify_save.py <output_file>

Shows summary statistics of quantized model including:
- Total tensor counts by type
- High-precision layers and their scale values
- Verification against expected layer patterns
"""

import sys
import argparse
from safetensors import safe_open
from convert_and_quantize.constants import (
    ZIMAGE_LAYER_KEYNAMES, QWEN_LAYER_KEYNAMES
)


def print_model_summary(output_file: str, model_type: str = "zimage"):
    """Print summary statistics of a quantized model."""
    
    with safe_open(output_file, framework='pt', device='cpu') as f:
        keys = f.keys()
        
        # Count tensor types
        weight_count = sum(1 for k in keys if k.endswith('.weight'))
        scale_count = sum(1 for k in keys if k.endswith('.scale_weight'))
        bias_count = sum(1 for k in keys if k.endswith('.bias'))
        other_count = len(keys) - weight_count - scale_count - bias_count
        
        print(f"{'='*60}")
        print(f"Model Summary: {output_file}")
        print(f"{'='*60}")
        print(f"\nTensor Counts:")
        print(f"  Total tensors: {len(keys)}")
        print(f"  .weight tensors: {weight_count}")
        print(f"  .scale_weight tensors: {scale_count}")
        print(f"  .bias tensors: {bias_count}")
        print(f"  Other tensors: {other_count}")
        
        # Identify high-precision layers (scale_weight = 1.0)
        scale_ones = []
        quantized_scales = []
        for k in sorted(keys):
            if '.scale_weight' in k:
                t = f.get_tensor(k)
                val = t.item()
                if val == 1.0:
                    scale_ones.append(k)
                else:
                    quantized_scales.append((k, val))
        
        print(f"\nHigh-Precision Analysis:")
        print(f"  High-precision layers (scale=1.0): {len(scale_ones)}")
        print(f"  Quantized layers (learned scale): {len(quantized_scales)}")
        
        # Show expected patterns and their counts
        if model_type == "zimage":
            layer_keynames = ZIMAGE_LAYER_KEYNAMES
        elif model_type == "qwen":
            layer_keynames = QWEN_LAYER_KEYNAMES
        else:
            layer_keynames = []
        
        if layer_keynames:
            print(f"\nExpected High-Precision Patterns ({model_type}):")
            for layer_name in layer_keynames:
                count = sum(1 for k in scale_ones if layer_name in k)
                print(f"  {layer_name}: {count} layers")
        
        # Show sample scale values
        print(f"\nSample Scale Values (first 10):")
        for k in sorted(keys):
            if '.scale_weight' in k:
                t = f.get_tensor(k)
                val = t.item() if t.numel() == 1 else 'array'
                dtype = str(t.dtype).split('.')[-1]
                print(f"  {k}: {val} ({dtype})")
                if sum(1 for _ in range(10)) >= 10:
                    break
        
        print(f"\n{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Verify and summarize a saved quantized model."
    )
    parser.add_argument(
        "model_file",
        help="Path to the saved .safetensors quantized model file"
    )
    parser.add_argument(
        "--model-type",
        type=str,
        default="zimage",
        choices=["zimage", "qwen"],
        help="Model type for pattern matching (default: zimage)"
    )
    
    args = parser.parse_args()
    
    try:
        print_model_summary(args.model_file, args.model_type)
    except FileNotFoundError:
        print(f"Error: File not found: {args.model_file}")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
