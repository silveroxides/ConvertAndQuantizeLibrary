# Installation Guide

## Prerequisites

- Python 3.8 or later
- PyTorch >= 1.13.0 with FP8 support (requires NVIDIA GPU or compatible hardware)
- pip or conda package manager

## Installation Methods

### 1. Basic Installation from PyPI (Recommended)

```bash
pip install convert-and-quantize
```

### 2. Installation with Optional Dependencies

#### ProdigyPlus Optimizer Support

ProdigyPlus is an advanced optimizer that can provide better quantization quality at the cost of additional computation.

```bash
pip install convert-and-quantize[prodigy]
```

#### Development Setup

For development and testing:

```bash
pip install convert-and-quantize[dev]
```

#### All Features

Install everything including optional dependencies:

```bash
pip install convert-and-quantize[all]
```

### 3. Installation from Source

Clone the repository and install in development mode:

```bash
git clone https://github.com/silveroxides/ConvertAndQuantizeLibrary.git
cd ConvertAndQuantizeLibrary
pip install -e .
```

For development:

```bash
pip install -e ".[dev]"
```

## Verification

Verify the installation by importing the package:

```python
import convert_and_quantize
print(convert_and_quantize.__version__)
```

Or run a quick test:

```python
import torch
from convert_and_quantize import LearnedRoundingConverter

converter = LearnedRoundingConverter()
weight = torch.randn(256, 256)
quantized, scale, dequantized = converter.convert(weight)
print("Quantization successful!")
```

## System Requirements

### Minimum Requirements

- 4GB RAM
- CUDA 11.0+ (for GPU acceleration, recommended)
- PyTorch with CUDA support

### Recommended Requirements

- 8GB+ RAM
- CUDA 12.0+
- GPU with at least 8GB VRAM for processing large models

## GPU Support

This library benefits significantly from GPU acceleration. To use GPU:

- Install CUDA Toolkit (version 11.0 or later)
- Install PyTorch with CUDA support:

```bash
# For CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

- Verify CUDA is available:

```python
import torch
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))
```

## Troubleshooting

### PyTorch Version Compatibility

If you encounter issues with FP8 support, ensure you have a compatible PyTorch version:

```python
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
```

### FP8 Support Check

Not all hardware supports FP8. Check if your system supports it:

```python
import torch
from convert_and_quantize import TARGET_FP8_DTYPE

try:
    test_tensor = torch.zeros(1, dtype=TARGET_FP8_DTYPE, device='cuda')
    print("FP8 is supported on your GPU")
except RuntimeError as e:
    print(f"FP8 not supported: {e}")
```

### CUDA Out of Memory

If you encounter CUDA memory errors:

1. Reduce `num_iter` in the converter
2. Use `scaling_mode="tensor"` instead of `"block"`
3. Process weights in smaller batches
4. Ensure no other GPU processes are running

### ProdigyPlus Installation Issues

If ProdigyPlus fails to install, it's optional and the library will work without it:

```python
from convert_and_quantize import get_optimizer

# This will fail gracefully if prodigyplus is not installed
try:
    optimizer = get_optimizer("ppsf")
except ImportError:
    print("ProdigyPlus not available, using 'adamw' instead")
    optimizer = get_optimizer("adamw")
```

## Updating

To update to the latest version:

```bash
pip install --upgrade convert-and-quantize
```

## Uninstallation

To remove the package:

```bash
pip uninstall convert-and-quantize
```

## Next Steps

After installation, check out:

- [Quick Start Guide](./QUICKSTART.md) for basic usage
- [Examples](../examples/) for practical code samples
- [API Reference](./API.md) for detailed documentation
