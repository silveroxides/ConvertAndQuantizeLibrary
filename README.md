# Convert and Quantize Library

Advanced quantization framework for neural networks - Convert models to FP8 with learned adaptive rounding and SVD-based optimization.

## Features

- **FP8 Quantization**: Convert neural network weights to FP8 format for reduced model size and faster inference
- **Learned Adaptive Rounding**: Advanced rounding strategy optimized using multiple algorithms (Original, AdamW, RAdam, ProdigyPlus)
- **SVD-Based Optimization**: Uses Singular Value Decomposition to identify and optimize the most important weight components
- **Model-Specific Presets**: Pre-configured layer exclusion/preservation lists for popular models (T5-XXL, Qwen, Hunyuan, Z-Image, WAN, NeRF, etc.)
- **Flexible Scaling**: Support for both tensor-level and block-level scaling modes
- **Bias Correction**: Automatic bias correction to maintain model accuracy after quantization
- **SafeTensors Support**: Direct integration with SafeTensors format for modern model distribution

## Installation

### Basic Installation
```bash
pip install convert-and-quantize
```

### With Optional Dependencies
```bash
# For ProdigyPlus optimizer support
pip install convert-and-quantize[prodigy]

# For development and testing
pip install convert-and-quantize[dev]

# All optional dependencies
pip install convert-and-quantize[all]
```

### Development Installation
```bash
git clone https://github.com/silveroxides/ConvertAndQuantizeLibrary
cd ConvertAndQuantizeLibrary
pip install -e ".[dev]"
```

## Quick Start

### Basic Usage

```python
import torch
from convert_and_quantize import LearnedRoundingConverter

# Initialize converter with default settings
converter = LearnedRoundingConverter(
    optimizer="original",
    num_iter=500,
    scaling_mode="tensor"
)

# Convert a weight tensor
weight = torch.randn(4096, 4096)
quantized, scale, dequantized = converter.convert(weight)

print(f"Original shape: {weight.shape}, dtype: {weight.dtype}")
print(f"Quantized shape: {quantized.shape}, dtype: {quantized.dtype}")
print(f"Scale: {scale.shape}")
```

### Convert Model from SafeTensors

```python
from safetensors import safe_open
from safetensors.torch import save_file
import torch
from convert_and_quantize import LearnedRoundingConverter

converter = LearnedRoundingConverter(
    optimizer="adamw",
    num_iter=1000,
    scaling_mode="block",
    block_size=64
)

# Load model
tensors = {}
with safe_open("model.safetensors", framework="pt", device="cpu") as f:
    for key in f.keys():
        tensors[key] = f.get_tensor(key)

# Quantize weights
new_tensors = {}
for key, tensor in tensors.items():
    if key.endswith(".weight") and tensor.ndim == 2:
        q_tensor, scale, _ = converter.convert(tensor)
        new_tensors[key] = q_tensor
        new_tensors[f"{key[:-7]}.scale_weight"] = scale
    else:
        new_tensors[key] = tensor

# Save quantized model
save_file(new_tensors, "model_fp8.safetensors")
```

## Configuration Options

### LearnedRoundingConverter Parameters

- **optimizer** (str): Optimizer choice - `"original"`, `"adamw"`, `"radam"`, or `"ppsf"`
- **num_iter** (int): Number of optimization iterations per tensor (default: 500)
- **top_p** (float): Proportion of principal components to use (default: 0.01)
- **min_k** (int): Minimum number of principal components (default: 1)
- **max_k** (int): Maximum number of principal components (default: 16)
- **scaling_mode** (str): `"tensor"` (per-tensor) or `"block"` (per-block) scaling
- **block_size** (int): Block size for block-level scaling (default: 64)
- **full_matrix** (bool): Use full SVD instead of low-rank approximation
- **lr** (float): Learning rate for optimizers (default: 0.01)

## Project Structure

```
convert_and_quantize/
├── __init__.py           # Package initialization and public API
├── constants.py          # Configuration constants and model-specific lists
├── core/
│   ├── __init__.py
│   └── converter.py      # Main LearnedRoundingConverter class
├── optimizers/
│   └── __init__.py       # Optimizer implementations
└── utils/
    └── __init__.py       # Utility functions
```

## API Reference

### Main Classes

#### `LearnedRoundingConverter`
Core quantization converter using learned adaptive rounding with SVD optimization.

```python
converter = LearnedRoundingConverter(
    optimizer="original",
    num_iter=500,
    top_p=0.01,
    scaling_mode="tensor"
)
quantized, scale, dequantized = converter.convert(weight_tensor)
```

### Utility Functions

#### `get_device()`
Get the appropriate device for computation.

```python
from convert_and_quantize import get_device
device = get_device()  # Returns 'cuda' or 'cpu'
```

#### `setup_seed()`
Set up reproducible random seed.

```python
from convert_and_quantize import setup_seed
generator = setup_seed(seed=42)
```

## Supported Models

Pre-configured layer lists for popular models:

- **T5-XXL**: Text encoder model from Google
- **Qwen**: Alibaba's language model series
- **Hunyuan**: Tencent's video diffusion model
- **Z-Image**: Custom image model
- **WAN**: Waymo autonomous driving models
- **NeRF**: Neural Radiance Field models
- **Radiance**: Radiance field rendering

## Performance Tips

1. **Use Block Scaling**: For very large tensors, block-level scaling can provide better accuracy
2. **Tune num_iter**: Start with 500 iterations and adjust based on results
3. **Choose Optimizer**: 
   - `"original"`: Fast, good for quick experiments
   - `"adamw"`: Stable, good general choice
   - `"radam"`: Robust, handles varying scales well
   - `"ppsf"`: High quality but requires ProdigyPlus installation
4. **Adjust top_p**: Lower values (0.001-0.01) for high-precision models, higher values for speed
5. **Use GPU**: Quantization is significantly faster on CUDA-enabled GPUs

## License

This project is licensed under the MIT License - see LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for bugs and feature requests.
