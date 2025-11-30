"""Runner to quantize and save SafeTensors models with learned rounding.

Usage:
  python scripts/quant_save_runner.py <model_path> [--num-iter N] [--exclude-layers FILTER] [--output OUTPUT]

This script loads a SafeTensors model, applies quantization with learned rounding,
and saves the quantized model along with scale weights and bias corrections.
Replicates the save pipeline from convert_fp8_svd_learn_w_blockwise.py.
"""
import argparse
import os
import sys
import torch

# Ensure repo root is on sys.path so local package imports work when running the script
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from safetensors.torch import safe_open, save_file
from convert_and_quantize import LearnedRoundingConverter
from convert_and_quantize.constants import (
    AVOID_KEY_NAMES, ZIMAGE_AVOID_KEY_NAMES, ZIMAGE_LAYER_KEYNAMES,
    QWEN_AVOID_KEY_NAMES, HUNYUAN_AVOID_KEY_NAMES
)
from convert_and_quantize.utils import generate_output_filename

# From convert_fp8_svd_learn_w_blockwise.py
TARGET_FP8_DTYPE = torch.float8_e4m3fn
COMPUTE_DTYPE = torch.float32
SCALE_DTYPE = torch.float32


def main():
    parser = argparse.ArgumentParser(description="Quantize and save a SafeTensors model with learned rounding.")
    parser.add_argument("model_path", help="Path to the .safetensors model file")
    parser.add_argument("--num-iter", type=int, default=50, help="Number of iterations for the converter (default: 50)")
    parser.add_argument(
        "--exclude-layers",
        type=str,
        default="zimage",
        help="Layer exclusion filter: 'zimage', 'qwen', 'hunyuan', or comma-separated list (default: zimage)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file path (default: auto-generated based on input and parameters)"
    )
    parser.add_argument(
        "--scaling-mode",
        type=str,
        default="tensor",
        choices=["tensor", "block"],
        help="Quantization scaling mode (default: tensor)"
    )
    parser.add_argument(
        "--min-k",
        type=int,
        default=256,
        help="Minimum number of components (default: 256)"
    )
    parser.add_argument(
        "--max-k",
        type=int,
        default=768,
        help="Maximum number of components (default: 768)"
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.1,
        help="Proportion of components to keep (default: 0.1)"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=8.0980000000000011e-3,
        help="Learning rate for optimization (default: 8.098e-3)"
    )

    args = parser.parse_args()

    model_path = args.model_path
    if not os.path.exists(model_path):
        print(f"Error: Model file not found: {model_path}")
        return

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
        # Parse comma-separated custom list
        avoid_keys = [k.strip() for k in exclude_filter.split(",") if k.strip()]
        layer_keys = []

    # Also include generic AVOID_KEY_NAMES
    all_avoid_keys = AVOID_KEY_NAMES + avoid_keys

    print(f"Loading model from: {model_path}")
    with safe_open(model_path, framework="pt", device="cpu") as f:
        tensors = {key: f.get_tensor(key) for key in f.keys()}

    print(f"Model loaded: {len(tensors)} tensors")

    # Filter 2D weight tensors
    weight_keys = [k for k, v in tensors.items() if k.endswith(".weight") and v.ndim == 2]
    print(f"Found {len(weight_keys)} 2D weight tensors to process")

    # Create converter
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    converter = LearnedRoundingConverter(
        optimizer="original",
        num_iter=args.num_iter,
        top_p=args.top_p,
        min_k=args.min_k,
        max_k=args.max_k,
        scaling_mode=args.scaling_mode,
    )

    # Initialize output tensors dict
    new_tensors = {}
    processed_count = 0
    skipped_count = 0
    total_weights = len(weight_keys)

    print("-" * 60)
    print("Starting quantization and saving process...")
    print("-" * 60)

    for i, key in enumerate(weight_keys):
        original_tensor = tensors[key]
        skip_reason = None
        create_scale_key = False
        process_this_key = True

        # Check if should be skipped entirely (AVOID list)
        if any(n in key for n in all_avoid_keys):
            skip_reason = "In avoid list"
            create_scale_key = False
            process_this_key = False

        # Check if should be kept in high precision (LAYER_KEYNAMES list)
        if any(n in key for n in layer_keys):
            skip_reason = "Keep in high precision"
            create_scale_key = True
            process_this_key = False

        if not process_this_key:
            if not create_scale_key:
                # Completely skip this tensor
                print(f"({i+1}/{total_weights}) Skipping: {key} ({skip_reason})")
                new_tensors[key] = original_tensor.to(device='cpu')
                skipped_count += 1
                continue
            else:
                # Keep in high precision with scale=1.0
                print(f"({i+1}/{total_weights}) High precision: {key} ({skip_reason})")
                new_tensors[key] = original_tensor.to(device='cpu')
                base_name = key[:key.rfind('.weight')]
                new_tensors[f"{base_name}.scale_weight"] = torch.tensor([1.0], dtype=SCALE_DTYPE).to(device='cpu')
                skipped_count += 1
                continue

        # Process: quantize the tensor
        print(f"({i+1}/{total_weights}) Quantizing: {key}")
        processed_count += 1

        # Skip empty or non-2D tensors (shouldn't happen given our filtering, but be safe)
        if original_tensor.numel() == 0 or original_tensor.ndim != 2:
            print(f"  - Skipping empty or non-2D tensor")
            new_tensors[key] = original_tensor.to(TARGET_FP8_DTYPE)
            base_name = key[:key.rfind('.weight')]
            new_tensors[f"{base_name}.scale_weight"] = torch.tensor([1.0], dtype=SCALE_DTYPE).to(device='cpu')
            continue

        # Quantize using converter
        try:
            q_tensor, dequant_s, dequant_w = converter.convert(original_tensor)
            new_tensors[key] = q_tensor.to(device='cpu')
            base_name = key[:key.rfind('.weight')]
            new_tensors[f"{base_name}.scale_weight"] = dequant_s.to(device='cpu').detach().clone()

            # Handle corresponding bias if it exists
            bias_key = f"{base_name}.bias"
            if bias_key in tensors:
                print(f"  - Adjusting bias: {bias_key}")
                original_bias = tensors[bias_key]
                new_tensors[bias_key] = original_bias.to(device='cpu')

            print(f"    - Scale shape: {dequant_s.shape}, Weight shape: {q_tensor.shape}")

        except Exception as e:
            print(f"  - ERROR: {e}")
            print(f"    - Skipping this tensor")
            new_tensors[key] = original_tensor.to(device='cpu')
            base_name = key[:key.rfind('.weight')]
            new_tensors[f"{base_name}.scale_weight"] = torch.tensor([1.0], dtype=SCALE_DTYPE).to(device='cpu')

        print("-" * 60)

    # Copy any remaining non-weight tensors (biases, embeddings, etc.)
    for key, tensor in tensors.items():
        if key not in new_tensors:
            new_tensors[key] = tensor.to(device='cpu')

    # Generate output filename if not provided
    if args.output:
        output_file = args.output
    else:
        output_file = generate_output_filename(
            model_path,
            TARGET_FP8_DTYPE,
            args.scaling_mode,
            min_k=args.min_k,
            max_k=args.max_k,
            top_p=args.top_p,
            lr=args.lr
        )

    print(f"\n{'='*60}")
    print(f"Quantization Summary:")
    print(f"  - Total weight tensors: {total_weights}")
    print(f"  - Quantized: {processed_count}")
    print(f"  - Skipped: {skipped_count}")
    print(f"  - Output tensors (with scales): {len(new_tensors)}")
    print(f"  - Output file: {output_file}")
    print(f"{'='*60}\n")

    # Save to output file
    try:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        print(f"Saving {len(new_tensors)} tensors to {output_file}")
        save_file(new_tensors, output_file)
        print("✓ Conversion and save complete!")
    except Exception as e:
        print(f"✗ FATAL: Error saving file '{output_file}': {e}")
        import traceback
        traceback.print_exc()
        return

    print(f"\nOutput saved to: {output_file}")


if __name__ == "__main__":
    main()
