"""Runner to test quantization on a SafeTensors model.

Usage:
  python scripts/quant_test_runner.py <model_path> [--num-iter N] [--exclude-layers FILTER]

This script will load 2D weight tensors from the provided SafeTensors
file, exclude layers based on the provided filter list, and run the
LearnedRoundingConverter with the requested parameters.
"""
import argparse
import os
import sys
import torch
# Ensure repo root is on sys.path so local package imports work when running the script
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from safetensors.torch import safe_open
from convert_and_quantize import LearnedRoundingConverter
from convert_and_quantize.constants import (
    AVOID_KEY_NAMES, ZIMAGE_AVOID_KEY_NAMES, ZIMAGE_LAYER_KEYNAMES,
    QWEN_AVOID_KEY_NAMES, HUNYUAN_AVOID_KEY_NAMES
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path", help="Path to the .safetensors model file")
    parser.add_argument("--num-iter", type=int, default=50, help="Number of iterations for the converter (default: 50)")
    parser.add_argument(
        "--exclude-layers",
        type=str,
        default="zimage",
        help="Layer exclusion filter: 'zimage', 'qwen', 'hunyuan', or comma-separated list (default: zimage)"
    )
    args = parser.parse_args()

    model_path = args.model_path
    num_iter = args.num_iter

    # Determine exclusion and high-precision filters
    exclude_filter = args.exclude_layers.lower().strip()
    if exclude_filter == "zimage":
        avoid_keys = ZIMAGE_AVOID_KEY_NAMES  # Skip entirely
        layer_keys = ZIMAGE_LAYER_KEYNAMES   # Keep in high precision (scale=1.0)
    elif exclude_filter == "qwen":
        avoid_keys = QWEN_AVOID_KEY_NAMES
        layer_keys = []
    elif exclude_filter == "hunyuan":
        avoid_keys = HUNYUAN_AVOID_KEY_NAMES
        layer_keys = []
    else:
        # Custom comma-separated list
        avoid_keys = [k.strip() for k in exclude_filter.split(",")]
        layer_keys = []

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    print(f"Using exclusion filter: {exclude_filter}")
    print(f"Layers to AVOID (skip entirely): {avoid_keys[:3]}..." if len(avoid_keys) > 3 else f"Layers to AVOID: {avoid_keys}")
    if layer_keys:
        print(f"Layers to keep in HIGH PRECISION (scale=1.0): {layer_keys[:3]}..." if len(layer_keys) > 3 else f"Layers for HIGH PRECISION: {layer_keys}")

    converter = LearnedRoundingConverter(
        optimizer="original",
        num_iter=num_iter,
        top_p=0.1,
        min_k=256,
        max_k=768,
        scaling_mode="tensor",
        lr=8.0980000000000011e-3,
    )

    print(f"Loading model tensors from: {model_path}")

    quantized_count = 0
    skipped_count = 0
    total_tensors = 0
    stats = []

    with safe_open(model_path, framework="pt", device="cpu") as f:
        keys = list(f.keys())
        print(f"Found {len(keys)} tensors in the file")

        for key in keys:
            total_tensors += 1
            try:
                tensor = f.get_tensor(key)
            except Exception as e:
                print(f"  - Could not read tensor {key}: {e}")
                skipped_count += 1
                continue

            if not key.endswith(".weight") or tensor.ndim != 2:
                # not a 2D weight matrix: skip
                skipped_count += 1
                continue

            # Check AVOID filter: skip entirely, no scale_weights
            should_avoid = False
            for avoid_key in avoid_keys:
                if avoid_key in key:
                    print(f"  - AVOIDING '{key}' (skip entirely, no scale_weights)")
                    should_avoid = True
                    break

            if should_avoid:
                skipped_count += 1
                continue

            # Check LAYER_KEYNAMES filter: keep in HIGH PRECISION (don't quantize)
            is_high_precision = False
            for layer_key in layer_keys:
                if layer_key in key:
                    print(f"  - KEEPING '{key}' in HIGH PRECISION (no quantization, original dtype)")
                    is_high_precision = True
                    break

            if is_high_precision:
                stats.append((key, 0.0, True))  # 0 error, marked as high precision
                quantized_count += 1
                skipped_count += 1  # Still count as processed but not quantized
                continue

            if tensor.numel() < 1000:
                print(f"  - Skipping small tensor '{key}' (numel={tensor.numel()})")
                skipped_count += 1
                continue

            print(f"  - Quantizing '{key}' (shape={list(tensor.shape)})...")
            try:
                t = tensor.to(device)
                q, scale, deq = converter.convert(t)
                
                # compute simple error metric
                error = (t.to(dtype=deq.dtype, device=deq.device) - deq).abs().mean().item()
                stats.append((key, error, False))
                quantized_count += 1
                print(f"    -> Mean abs error: {error:.6e}")
            except Exception as e:
                print(f"    -> Error quantizing '{key}': {e}")
                skipped_count += 1

    print("\nQuantization run complete")
    print(f"  Total tensors seen: {total_tensors}")
    print(f"  Quantized tensors: {quantized_count}")
    print(f"  Skipped tensors: {skipped_count}")

    if stats:
        mean_error = sum(s[1] for s in stats) / len(stats)
        high_precision_count = sum(1 for s in stats if s[2])
        print(f"\nSummary Mean Absolute Error across quantized tensors: {mean_error:.6e}")
        print(f"High precision (scale=1.0) tensors: {high_precision_count}")



if __name__ == "__main__":
    main()
