"""Runner to quantize and save SafeTensors models with learned rounding.

Usage:
  python scripts/quant_save_runner.py <model_path> [--num-iter N] [--exclude-layers FILTER] [--output OUTPUT]

This script is a thin wrapper around the `quantize_and_save` function.
"""
import argparse
import os
import sys

# Ensure repo root is on sys.path so local package imports work when running the script
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from convert_and_quantize.cli import quantize_and_save


def main():
    parser = argparse.ArgumentParser(description="Quantize and save a SafeTensors model with learned rounding.")
    parser.add_argument("model_path", help="Path to the .safetensors model file")
    parser.add_argument("--num-iter", type=int, default=50, help="Number of iterations for the converter (default: 50)")
    parser.add_argument(
        "--exclude-layers",
        type=str,
        default="zimage",
        help="""Layer exclusion filter. Predefined: 'zimage', 'qwen', 'hunyuan', 'chroma_l', 'chroma_s', 
            'nerf_l', 'nerf_s', 'radiance', 'wan', or comma-separated custom list (default: zimage)"""
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

    quantize_and_save(
        model_path=args.model_path,
        num_iter=args.num_iter,
        exclude_layers=args.exclude_layers,
        output=args.output,
        scaling_mode=args.scaling_mode,
        min_k=args.min_k,
        max_k=args.max_k,
        top_p=args.top_p,
        lr=args.lr,
    )


if __name__ == "__main__":
    main()

