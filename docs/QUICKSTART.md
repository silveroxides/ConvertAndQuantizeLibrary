# Quick Start Guide

Get started with Convert and Quantize in 5 minutes!

## Installation

```bash
pip install convert-and-quantize
```

## Your First Quantization

### 1. Import and Initialize

```python
import torch
from convert_and_quantize import LearnedRoundingConverter

# Create a converter with default settings
converter = LearnedRoundingConverter(
    optimizer="original",
    num_iter=500
)
```

### 2. Quantize a Tensor

```python
# Create a sample weight tensor
weight = torch.randn(4096, 4096)

# Quantize it
quantized, scale, dequantized = converter.convert(weight)

print(f"Original dtype: {weight.dtype}")
print(f"Quantized dtype: {quantized.dtype}")
print(f"Memory saved: {(1 - quantized.element_size()/weight.element_size()) * 100:.1f}%")
```

### 3. Check Quality

```python
# Calculate error
error = (weight - dequantized).abs().mean()
print(f"Quantization error: {error:.6e}")
```

## Common Use Cases

### Use Different Optimizer

```python
# Try a different optimization algorithm
converter = LearnedRoundingConverter(
    optimizer="adamw",  # or "radam", "ppsf"
    num_iter=1000  # More iterations for better quality
)
```

### Block-Level Quantization

For large tensors, use block-level scaling for better accuracy:

```python
converter = LearnedRoundingConverter(
    scaling_mode="block",
    block_size=64,
    num_iter=500
)
```

### Batch Quantization

```python
# Quantize multiple tensors
weights = [
    torch.randn(4096, 4096),
    torch.randn(2048, 8192),
    torch.randn(1024, 2048),
]

converter = LearnedRoundingConverter(num_iter=300)

for i, weight in enumerate(weights):
    quantized, scale, dequantized = converter.convert(weight)
    print(f"Tensor {i}: shape={quantized.shape}, error={...}")
```

### Use with SafeTensors

```python
from safetensors import safe_open
from safetensors.torch import save_file

# Load model
tensors = {}
with safe_open("model.safetensors", framework="pt", device="cpu") as f:
    for key in f.keys():
        tensors[key] = f.get_tensor(key)

# Quantize
converter = LearnedRoundingConverter(num_iter=500)
new_tensors = {}

for key, tensor in tensors.items():
    if key.endswith(".weight") and tensor.ndim == 2:
        q_tensor, scale, _ = converter.convert(tensor)
        new_tensors[key] = q_tensor
        new_tensors[f"{key[:-7]}.scale_weight"] = scale
    else:
        new_tensors[key] = tensor

# Save
save_file(new_tensors, "model_fp8.safetensors")
```

## Configuration Tips

### For Speed (Quick Prototyping)

```python
converter = LearnedRoundingConverter(
    optimizer="original",
    num_iter=100,
    top_p=0.05  # Use fewer SVD components
)
```

### For Quality (Production)

```python
converter = LearnedRoundingConverter(
    optimizer="adamw",  # More stable
    num_iter=1000,
    top_p=0.01,  # Use more SVD components
    scaling_mode="block",
    block_size=32
)
```

### Balanced (Recommended Default)

```python
converter = LearnedRoundingConverter(
    optimizer="adamw",
    num_iter=500,
    top_p=0.01,
    scaling_mode="tensor"
)
```

## Utility Functions

### Check Device

```python
from convert_and_quantize import get_device

device = get_device()  # 'cuda' or 'cpu'
print(f"Using device: {device}")
```

### Setup Seed for Reproducibility

```python
from convert_and_quantize import setup_seed

generator = setup_seed(seed=-1)
```

### Get Constants

```python
from convert_and_quantize import FP8_MIN, FP8_MAX

print(f"FP8 range: [{FP8_MIN}, {FP8_MAX}]")
```

## Troubleshooting

### CUDA Out of Memory

```python
# Reduce iterations or switch to tensor scaling
converter = LearnedRoundingConverter(
    num_iter=100,  # Fewer iterations
    scaling_mode="tensor"  # Not block
)
```

### Poor Quantization Quality

```python
# Increase iterations and use better optimizer
converter = LearnedRoundingConverter(
    optimizer="adamw",
    num_iter=2000,
    top_p=0.02
)
```

### Slow Quantization

```python
# Make it faster
converter = LearnedRoundingConverter(
    optimizer="original",
    num_iter=200,
    top_p=0.05
)
```

## Next Steps

- Explore [Examples](../examples/) for more advanced usage
- Read the [API Reference](./API.md) for detailed documentation
- Check out the [Installation Guide](./INSTALLATION.md) for different install options

## Getting Help

- Check the [GitHub Issues](https://github.com/silveroxides/ConvertAndQuantizeLibrary/issues)
- Read the examples in the `examples/` directory
- Review the full documentation

Happy quantizing!
