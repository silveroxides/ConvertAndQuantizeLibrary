"""
Example 1: Basic tensor quantization

This example demonstrates how to use the LearnedRoundingConverter to quantize
a single weight tensor to FP8 format.
"""

import torch
from convert_and_quantize import LearnedRoundingConverter


def example_basic_quantization():
    """Basic quantization of a single tensor."""
    print("=" * 60)
    print("Example 1: Basic Tensor Quantization")
    print("=" * 60)

    # Create a sample weight tensor
    weight = torch.randn(4096, 4096, dtype=torch.float32)
    print(f"Original weight: shape={weight.shape}, dtype={weight.dtype}")

    # Initialize converter
    converter = LearnedRoundingConverter(
        optimizer="original",
        num_iter=100,  # Reduced for example
        scaling_mode="tensor"
    )

    # Quantize
    print("\nQuantizing tensor...")
    quantized, scale, dequantized = converter.convert(weight)

    print(f"\nQuantized weight: shape={quantized.shape}, dtype={quantized.dtype}")
    print(f"Scale factor: {scale}")
    print(f"Dequantized weight: shape={dequantized.shape}, dtype={dequantized.dtype}")

    # Calculate error
    error = (weight - dequantized).abs().mean()
    max_error = (weight - dequantized).abs().max()
    print(f"\nQuantization Error:")
    print(f"  Mean Absolute Error: {error:.6e}")
    print(f"  Max Absolute Error:  {max_error:.6e}")

    # Memory efficiency
    original_size_mb = weight.element_size() * weight.numel() / (1024 ** 2)
    quantized_size_mb = quantized.element_size() * quantized.numel() / (1024 ** 2)
    print(f"\nMemory Usage:")
    print(f"  Original: {original_size_mb:.2f} MB")
    print(f"  Quantized: {quantized_size_mb:.2f} MB")
    print(f"  Reduction: {(1 - quantized_size_mb/original_size_mb) * 100:.1f}%")


if __name__ == "__main__":
    example_basic_quantization()
