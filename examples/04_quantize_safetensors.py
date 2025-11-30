"""
Example 4: Quantizing a model from SafeTensors

This example demonstrates how to quantize an entire model stored in
SafeTensors format, which is commonly used for distributing large models.
"""

import torch
from safetensors.torch import save_file
from convert_and_quantize import LearnedRoundingConverter, SCALE_DTYPE


def create_dummy_model_tensors(num_layers=4):
    """Create dummy model tensors for demonstration."""
    tensors = {}
    
    for i in range(num_layers):
        # Attention weights
        tensors[f"layers.{i}.attention.q_proj.weight"] = torch.randn(4096, 4096)
        tensors[f"layers.{i}.attention.k_proj.weight"] = torch.randn(4096, 4096)
        tensors[f"layers.{i}.attention.v_proj.weight"] = torch.randn(4096, 4096)
        tensors[f"layers.{i}.attention.out_proj.weight"] = torch.randn(4096, 4096)
        
        # MLP weights
        tensors[f"layers.{i}.mlp.fc1.weight"] = torch.randn(16384, 4096)
        tensors[f"layers.{i}.mlp.fc2.weight"] = torch.randn(4096, 16384)
        
        # Normalization parameters (usually not quantized)
        tensors[f"layers.{i}.attention.norm.weight"] = torch.randn(4096)
        tensors[f"layers.{i}.mlp.norm.weight"] = torch.randn(4096)
    
    # Embeddings (usually not quantized)
    tensors["token_embedding.weight"] = torch.randn(50000, 4096)
    tensors["position_embedding.weight"] = torch.randn(2048, 4096)
    
    return tensors


def example_quantize_safetensors_model():
    """Quantize a model from SafeTensors format."""
    print("=" * 60)
    print("Example 4: Quantizing a Model from SafeTensors")
    print("=" * 60)

    # Create dummy model tensors
    print("\nCreating dummy model tensors...")
    tensors = create_dummy_model_tensors(num_layers=4)
    print(f"Created {len(tensors)} tensors")

    # Initialize converter
    converter = LearnedRoundingConverter(
        optimizer="adamw",
        num_iter=100,
        scaling_mode="tensor"
    )

    # Quantize weights
    print("\nQuantizing weight tensors...")
    new_tensors = {}
    weight_keys = [k for k in tensors.keys() if k.endswith(".weight") and tensors[k].ndim == 2]
    
    for i, key in enumerate(weight_keys):
        tensor = tensors[key]
        
        # Skip small parameters
        if tensor.numel() < 1000:
            print(f"  [{i+1}/{len(weight_keys)}] Skipping {key} (too small)")
            new_tensors[key] = tensor
            continue
        
        print(f"  [{i+1}/{len(weight_keys)}] Quantizing {key}...")
        
        # Quantize
        q_tensor, scale, _ = converter.convert(tensor)
        
        # Store quantized weight and scale
        new_tensors[key] = q_tensor
        base_name = key[:-7]  # Remove ".weight"
        new_tensors[f"{base_name}.scale_weight"] = scale.to(dtype=SCALE_DTYPE)
    
    # Keep non-weight parameters
    print("\nKeeping non-weight parameters...")
    for key, tensor in tensors.items():
        if key not in new_tensors:
            new_tensors[key] = tensor

    # Summary
    print("\n" + "=" * 60)
    print("Quantization Summary:")
    print(f"  Original tensor count: {len(tensors)}")
    print(f"  Quantized tensor count: {len(new_tensors)}")
    
    original_size = sum(t.numel() * t.element_size() for t in tensors.values()) / (1024**2)
    quantized_size = sum(t.numel() * t.element_size() for t in new_tensors.values()) / (1024**2)
    
    print(f"  Original size: {original_size:.2f} MB")
    print(f"  Quantized size: {quantized_size:.2f} MB")
    print(f"  Reduction: {(1 - quantized_size/original_size) * 100:.1f}%")

    # Save example (commented out to avoid file creation in examples)
    # output_file = "model_fp8.safetensors"
    # print(f"\nSaving to {output_file}...")
    # save_file(new_tensors, output_file)
    # print("Done!")


if __name__ == "__main__":
    example_quantize_safetensors_model()
