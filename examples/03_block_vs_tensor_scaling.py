"""
Example 3: Block-level quantization for large models

This example demonstrates block-level quantization which can provide
better accuracy for very large tensors compared to tensor-level quantization.
"""

import torch
from convert_and_quantize import LearnedRoundingConverter


def example_block_vs_tensor_scaling():
    """Compare block-level and tensor-level quantization."""
    print("=" * 60)
    print("Example 3: Block vs Tensor Scaling")
    print("=" * 60)

    # Create a large weight tensor
    weight = torch.randn(8192, 4096, dtype=torch.float32)
    print(f"\nOriginal weight: shape={weight.shape}, dtype={weight.dtype}")

    # Test tensor-level scaling
    print("\n--- Tensor-level Scaling ---")
    converter_tensor = LearnedRoundingConverter(
        optimizer="adamw",
        num_iter=150,
        scaling_mode="tensor"
    )

    quantized_t, scale_t, dequantized_t = converter_tensor.convert(weight.clone())
    error_t = (weight - dequantized_t).abs().mean()
    max_error_t = (weight - dequantized_t).abs().max()

    print(f"Scale shape: {scale_t.shape}")
    print(f"Mean error: {error_t:.6e}")
    print(f"Max error:  {max_error_t:.6e}")

    # Test block-level scaling
    print("\n--- Block-level Scaling (64-block size) ---")
    converter_block = LearnedRoundingConverter(
        optimizer="adamw",
        num_iter=150,
        scaling_mode="block",
        block_size=64
    )

    quantized_b, scale_b, dequantized_b = converter_block.convert(weight.clone())
    error_b = (weight - dequantized_b).abs().mean()
    max_error_b = (weight - dequantized_b).abs().max()

    print(f"Scale shape: {scale_b.shape}")
    print(f"Mean error: {error_b:.6e}")
    print(f"Max error:  {max_error_b:.6e}")

    # Comparison
    print("\n" + "=" * 60)
    print("Comparison:")
    print(f"Tensor-level - Mean: {error_t:.6e}, Max: {max_error_t:.6e}")
    print(f"Block-level  - Mean: {error_b:.6e}, Max: {max_error_b:.6e}")
    improvement = ((error_t - error_b) / error_t) * 100
    print(f"Improvement with block scaling: {improvement:.1f}%")


if __name__ == "__main__":
    example_block_vs_tensor_scaling()
