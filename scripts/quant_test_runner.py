"""Runner to test quantization on a SafeTensors model.

Usage:
  python scripts/quant_test_runner.py <model_path> [--num-iter N]

This script will load 2D weight tensors from the provided SafeTensors
file, exclude any layers containing the substring 'zimage_l', and run the
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path", help="Path to the .safetensors model file")
    parser.add_argument("--num-iter", type=int, default=100, help="Number of iterations for the converter (default: 100)")
    args = parser.parse_args()

    model_path = args.model_path
    num_iter = args.num_iter

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    converter = LearnedRoundingConverter(
        optimizer="original",
        num_iter=num_iter,
        top_p=0.1,
        min_k=256,
        max_k=768,
        scaling_mode="tensor",
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

            if "zimage_l" in key:
                print(f"  - Excluding layer '{key}' (matches exclusion filter 'zimage_l')")
                skipped_count += 1
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
                stats.append((key, error))
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
        mean_error = sum(s for _, s in stats) / len(stats)
        print(f"\nSummary Mean Absolute Error across quantized tensors: {mean_error:.6e}")


if __name__ == "__main__":
    main()
