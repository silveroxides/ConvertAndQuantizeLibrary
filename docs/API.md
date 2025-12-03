# API Reference

Complete API documentation for the Convert and Quantize library.

## Table of Contents

- [Core Classes](#core-classes)
- [Utility Functions](#utility-functions)
- [Constants](#constants)
- [Optimizers](#optimizers)

## Core Classes

### LearnedRoundingConverter

Main quantization converter using learned adaptive rounding with SVD optimization.

#### Constructor

```python
LearnedRoundingConverter(
    optimizer="original",
    num_iter=500,
    top_p=0.01,
    min_k=1,
    max_k=16,
    scaling_mode='tensor',
    block_size=64,
    full_matrix=False,
    **kwargs
)
```

**Parameters:**

- `optimizer` (str): Optimization algorithm to use
  - `"original"`: Adaptive learning rate with dynamic scheduling (default)
  - `"adamw"`: AdamW optimizer
  - `"radam"`: RAdam optimizer
  - `"ppsf"`: ProdigyPlus optimizer (requires `prodigyplus` package)

- `num_iter` (int, default=500): Number of optimization iterations per tensor
  - Higher values generally produce better results but take longer
  - Typical range: 100-2000

- `top_p` (float, default=0.01): Proportion of principal components to use
  - Lower values (0.001-0.01) for high-precision models
  - Higher values (0.02-0.1) for faster optimization
  - Must be between 0 and 1

- `min_k` (int, default=1): Minimum number of SVD components
  - Ensures we use at least this many components
  - Must be >= 1

- `max_k` (int, default=16): Maximum number of SVD components
  - Limits the number of components used
  - Must be >= min_k

- `scaling_mode` (str, default='tensor'): Scaling strategy
  - `'tensor'`: Single scale factor per tensor
  - `'block'`: Separate scale factor per block (more memory intensive but better accuracy)

- `block_size` (int, default=64): Block size for block-level scaling
  - Only used when `scaling_mode='block'`
  - Typical values: 32, 64, 128

- `full_matrix` (bool, default=False): Use full SVD computation
  - `False`: Use low-rank SVD approximation (faster)
  - `True`: Use full SVD (slower but more accurate)

- `**kwargs`: Additional optimizer-specific parameters
  - `lr` (float, default=0.01): Learning rate for optimizers
  - Other optimizer-specific parameters passed to the optimization function

#### Methods

##### convert()

Convert a weight tensor to quantized FP8 format.

```python
quantized, scale, dequantized = converter.convert(W_orig)
```

**Parameters:**

- `W_orig` (torch.Tensor): Original weight tensor to quantize

**Returns:**

- `quantized` (torch.Tensor): Quantized tensor in FP8 format
- `scale` (torch.Tensor): Scale factors used for quantization
  - Shape: `(1,)` for tensor-level scaling
  - Shape: `(out_features, num_blocks, 1)` for block-level scaling
- `dequantized` (torch.Tensor): Dequantized tensor for verification

**Example:**

```python
import torch
from convert_and_quantize import LearnedRoundingConverter

converter = LearnedRoundingConverter(num_iter=500)
weight = torch.randn(4096, 4096)

quantized, scale, dequantized = converter.convert(weight)

# Check quantization error
error = (weight - dequantized).abs().mean()
print(f"Error: {error:.6e}")
```

## Utility Functions

### get_device()

Get the appropriate device for computation.

```python
device = get_device()
```

**Returns:**

- `str`: Either `'cuda'` if available, else `'cpu'`

**Example:**

```python
from convert_and_quantize import get_device

device = get_device()
tensor = torch.randn(10, 10, device=device)
```

### get_fp8_constants()

Get FP8 data type constants (min, max, min_positive).

```python
min_val, max_val, min_pos = get_fp8_constants(fp8_dtype)
```

**Parameters:**

- `fp8_dtype` (torch.dtype): FP8 data type

**Returns:**

- `tuple`: (min_value, max_value, min_positive_value)

**Example:**

```python
from convert_and_quantize import get_fp8_constants, TARGET_FP8_DTYPE

min_val, max_val, min_pos = get_fp8_constants(TARGET_FP8_DTYPE)
print(f"FP8 range: [{min_val}, {max_val}]")
print(f"Min positive: {min_pos}")
```

### clean_gpu_memory()

Clean up GPU memory by running garbage collection and emptying CUDA cache.

```python
clean_gpu_memory(device=None)
```

**Parameters:**

- `device` (str, optional): Device to clean. If None, current device is used.

**Example:**

```python
from convert_and_quantize import clean_gpu_memory

# Process large batch of tensors
for tensor in large_tensor_list:
    process(tensor)

# Clean up memory after batch
clean_gpu_memory()
```

### setup_seed()

Set up reproducible random seed for reproducibility.

```python
generator = setup_seed(seed, device=None)
```

**Parameters:**

- `seed` (int): Random seed value
- `device` (str, optional): Device for generator. If None, uses current device.

**Returns:**

- `torch.Generator`: Generator object with set seed

**Example:**

```python
from convert_and_quantize import setup_seed

gen = setup_seed(-1)
random_tensor = torch.randn(10, 10, generator=gen)
```

### should_process_layer()

Determine if a layer should be quantized based on model type and layer name.

```python
should_process, create_scale, reason = should_process_layer(
    key,
    t5xxl=False,
    keep_distillation_large=False,
    keep_distillation_small=False,
    keep_nerf_large=False,
    keep_nerf_small=False,
    radiance=False,
    wan=False,
    qwen=False,
    hunyuan=False,
    zimage_l=False,
    zimage_s=False
)
```

**Parameters:**

- `key` (str): Layer key/name
- `t5xxl` (bool): Apply T5-XXL model exclusions
- `keep_distillation_large` (bool): Keep large distillation layers
- `keep_distillation_small` (bool): Keep small distillation layers
- `keep_nerf_large` (bool): Keep large NeRF layers
- `keep_nerf_small` (bool): Keep small NeRF layers
- `radiance` (bool): Apply Radiance field exclusions
- `wan` (bool): Apply WAN model exclusions
- `qwen` (bool): Apply Qwen model exclusions
- `hunyuan` (bool): Apply Hunyuan model exclusions
- `zimage_l` (bool): Apply Z-Image large model exclusions
- `zimage_s` (bool): Apply Z-Image small model exclusions

**Returns:**

- `bool`: Whether to process this layer
- `bool`: Whether to create scale factor
- `str`: Reason for decision

**Example:**

```python
from convert_and_quantize import should_process_layer

key = "attention.q_proj.weight"
should_process, create_scale, reason = should_process_layer(
    key, t5xxl=True
)

if should_process:
    # Quantize this layer
    pass
```

### generate_output_filename()

Generate output filename based on quantization configuration.

```python
output_file = generate_output_filename(
    input_file,
    target_dtype,
    scaling_mode,
    **kwargs
)
```

**Parameters:**

- `input_file` (str): Input file path
- `target_dtype` (torch.dtype): Target FP8 dtype
- `scaling_mode` (str): Scaling mode ('tensor' or 'block')
- Additional parameters for filename generation

**Returns:**

- `str`: Generated output filename

**Example:**

```python
from convert_and_quantize import generate_output_filename, TARGET_FP8_DTYPE

output = generate_output_filename(
    "model.safetensors",
    TARGET_FP8_DTYPE,
    "block",
    min_k=1,
    max_k=16,
    top_p=0.01
)
print(output)  # e.g., "model_float8_e4m3fn_block_k1-16_p0.01.safetensors"
```

## Constants

### Data Type Constants

```python
from convert_and_quantize import (
    TARGET_FP8_DTYPE,
    COMPUTE_DTYPE,
    SCALE_DTYPE,
    FP8_MIN,
    FP8_MAX,
    FP8_MIN_POS,
)
```

- `TARGET_FP8_DTYPE` (torch.dtype): Target FP8 data type (e4m3fn)
- `COMPUTE_DTYPE` (torch.dtype): Data type for computation (float32)
- `SCALE_DTYPE` (torch.dtype): Data type for scale factors (float32)
- `FP8_MIN` (float): Minimum representable value in FP8
- `FP8_MAX` (float): Maximum representable value in FP8
- `FP8_MIN_POS` (float): Minimum positive representable value in FP8

### Model-Specific Constants

```python
from convert_and_quantize.constants import (
    AVOID_KEY_NAMES,
    T5XXL_REMOVE_KEY_NAMES,
    QWEN_AVOID_KEY_NAMES,
    HUNYUAN_AVOID_KEY_NAMES,
    ZIMAGE_AVOID_KEY_NAMES,
    RADIANCE_LAYER_KEYNAMES,
    WAN_LAYER_KEYNAMES,
    QWEN_LAYER_KEYNAMES,
    ZIMAGE_LAYER_KEYNAMES,
    DISTILL_LAYER_KEYNAMES_LARGE,
    DISTILL_LAYER_KEYNAMES_SMALL,
    NERF_LAYER_KEYNAMES_LARGE,
    NERF_LAYER_KEYNAMES_SMALL,
)
```

These constants define layer names to exclude or preserve for specific models.

## Optimizers

### get_optimizer()

Get an optimizer function by name.

```python
from convert_and_quantize import get_optimizer

optimizer_func = get_optimizer("adamw")
```

**Parameters:**

- `name` (str): Optimizer name
  - `"original"`: Adaptive learning rate optimization
  - `"adamw"`: AdamW optimizer
  - `"radam"`: RAdam optimizer
  - `"ppsf"`: ProdigyPlus optimizer

**Returns:**

- `Callable`: Optimizer function

**Raises:**

- `ValueError`: If optimizer name is not recognized

**Example:**

```python
from convert_and_quantize.optimizers import get_optimizer

optimizer = get_optimizer("adamw")
# Use in custom quantization workflow
```

## Complete Example

```python
import torch
from safetensors.torch import save_file
from convert_and_quantize import (
    LearnedRoundingConverter,
    get_device,
    setup_seed,
    SCALE_DTYPE,
    TARGET_FP8_DTYPE,
)

# Setup
device = get_device()
generator = setup_seed(-1)

# Create converter
converter = LearnedRoundingConverter(
    optimizer="adamw",
    num_iter=1000,
    scaling_mode="block",
    block_size=64
)

# Quantize model
weights = {
    "layer1.weight": torch.randn(4096, 4096),
    "layer2.weight": torch.randn(2048, 8192),
}

quantized_weights = {}
for key, weight in weights.items():
    q, scale, _ = converter.convert(weight)
    quantized_weights[key] = q
    quantized_weights[f"{key[:-7]}.scale"] = scale.to(SCALE_DTYPE)

# Save
save_file(quantized_weights, "quantized.safetensors")
```
