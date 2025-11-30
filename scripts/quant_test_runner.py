"""Runner to test quantization on a SafeTensors model.

Usage:
  python scripts/quant_test_runner.py <model_path> [--num-iter N] [--exclude-layers FILTER]

This script is a thin wrapper around the `test_quantization` function.
"""
import argparse
import os
import sys

# Ensure repo root is on sys.path so local package imports work when running the script
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from convert_and_quantize.cli import test_quantization


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path", help="Path to the .safetensors model file")
    parser.add_argument("--num-iter", type=int, default=50, help="Number of iterations for the converter (default: 50)")
    parser.add_argument(
        "--exclude-layers",
        type=str,
        default="zimage",
        help="""Layer exclusion filter. Predefined: 'zimage', 'qwen', 'hunyuan', 'chroma_l', 'chroma_s', 
            'nerf_l', 'nerf_s', 'radiance', 'wan', or comma-separated custom list (default: zimage)"""
    )
    args = parser.parse_args()

    test_quantization(
        model_path=args.model_path,
        num_iter=args.num_iter,
        exclude_layers=args.exclude_layers,
    )


if __name__ == "__main__":
    main()
