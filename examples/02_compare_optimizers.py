"""
Example 2: Batch quantization with different optimizers

This example shows how to compare different optimization algorithms
for quantizing multiple tensors.
"""

import torch
from convert_and_quantize import LearnedRoundingConverter


def quantize_with_optimizer(optimizer_name, tensors):
    """Quantize tensors using a specific optimizer."""
    print(f"\n--- Using {optimizer_name.upper()} Optimizer ---")

    converter = LearnedRoundingConverter(
        optimizer=optimizer_name,
        num_iter=200,
        scaling_mode="tensor"
    )

    total_error = 0
    for i, weight in enumerate(tensors):
        quantized, scale, dequantized = converter.convert(weight)
        error = (weight - dequantized).abs().mean()
        total_error += error.item()
        print(f"  Tensor {i+1}: error = {error:.6e}")

    avg_error = total_error / len(tensors)
    print(f"  Average Error: {avg_error:.6e}")
    return avg_error


def example_compare_optimizers():
    """Compare different optimizers for quantization."""
    print("=" * 60)
    print("Example 2: Compare Quantization Optimizers")
    print("=" * 60)

    # Create sample tensors
    tensors = [
        torch.randn(2048, 2048, dtype=torch.float32),
        torch.randn(4096, 1024, dtype=torch.float32),
        torch.randn(512, 8192, dtype=torch.float32),
    ]

    print(f"\nQuantizing {len(tensors)} tensors with different optimizers...")

    results = {}
    for optimizer in ["original", "adamw", "radam"]:
        try:
            error = quantize_with_optimizer(optimizer, tensors)
            results[optimizer] = error
        except Exception as e:
            print(f"  Error with {optimizer}: {e}")

    print("\n" + "=" * 60)
    print("Summary:")
    for optimizer, error in sorted(results.items(), key=lambda x: x[1]):
        print(f"  {optimizer:10s}: {error:.6e}")


if __name__ == "__main__":
    example_compare_optimizers()
