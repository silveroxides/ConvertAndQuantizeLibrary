"""
Utility functions for quantization operations.
"""

import torch
import gc
import os
from typing import Tuple, Dict, Optional, Union


def get_device() -> str:
    """
    Get the appropriate device for computation.
    
    Returns:
        'cuda' if available, else 'cpu'
    """
    return 'cuda' if torch.cuda.is_available() else 'cpu'


def get_fp8_constants(fp8_dtype: torch.dtype) -> Tuple[float, float, float]:
    """
    Get FP8 data type constants.
    
    Args:
        fp8_dtype: FP8 dtype to query
        
    Returns:
        Tuple of (min, max, min_positive)
    """
    finfo = torch.finfo(fp8_dtype)
    return float(finfo.min), float(finfo.max), float(finfo.tiny)


def clean_gpu_memory(device: Optional[str] = None):
    """
    Clean up GPU memory.
    
    Args:
        device: Device to clean. If None, checks current device.
    """
    gc.collect()
    if device is None:
        device = get_device()
    if device == 'cuda':
        torch.cuda.empty_cache()


def setup_seed(seed: int, device: Optional[str] = None) -> torch.Generator:
    """
    Set up reproducible random seed.
    
    Args:
        seed: Random seed value
        device: Device for generator
        
    Returns:
        Generator object
    """
    if device is None:
        device = get_device()
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    return generator


def get_layer_filters(filter_name: str) -> Tuple[list, list]:
    """
    Get layer exclusion and preservation filters based on a filter name.

    Args:
        filter_name: Predefined filter name or custom comma-separated list.

    Returns:
        Tuple of (all_avoid_keys, layer_keys)
    """
    from convert_and_quantize.constants import (
        AVOID_KEY_NAMES, ZIMAGE_AVOID_KEY_NAMES, ZIMAGE_LAYER_KEYNAMES,
        QWEN_AVOID_KEY_NAMES, QWEN_LAYER_KEYNAMES, HUNYUAN_AVOID_KEY_NAMES,
        DISTILL_LAYER_KEYNAMES_LARGE, DISTILL_LAYER_KEYNAMES_SMALL,
        NERF_LAYER_KEYNAMES_LARGE, NERF_LAYER_KEYNAMES_SMALL,
        RADIANCE_AVOID_KEY_NAMES, WAN_LAYER_KEYNAMES
    )

    filter_name = filter_name.lower().strip()

    if filter_name == "zimage":
        layer_keys = ZIMAGE_LAYER_KEYNAMES
        all_avoid_keys = ZIMAGE_AVOID_KEY_NAMES + AVOID_KEY_NAMES
    elif filter_name == "qwen":
        layer_keys = QWEN_LAYER_KEYNAMES
        all_avoid_keys = QWEN_AVOID_KEY_NAMES + AVOID_KEY_NAMES
    elif filter_name == "hunyuan":
        layer_keys = []
        all_avoid_keys = HUNYUAN_AVOID_KEY_NAMES + AVOID_KEY_NAMES
    elif filter_name == "chroma_l":
        layer_keys = DISTILL_LAYER_KEYNAMES_LARGE
        all_avoid_keys = AVOID_KEY_NAMES
    elif filter_name == "chroma_s":
        layer_keys = DISTILL_LAYER_KEYNAMES_SMALL
        all_avoid_keys = AVOID_KEY_NAMES
    elif filter_name == "nerf_l":
        layer_keys = NERF_LAYER_KEYNAMES_LARGE
        all_avoid_keys = AVOID_KEY_NAMES + RADIANCE_AVOID_KEY_NAMES
    elif filter_name == "nerf_s":
        layer_keys = NERF_LAYER_KEYNAMES_SMALL
        all_avoid_keys = AVOID_KEY_NAMES + RADIANCE_AVOID_KEY_NAMES
    elif filter_name == "radiance":
        layer_keys = []
        all_avoid_keys = AVOID_KEY_NAMES + RADIANCE_AVOID_KEY_NAMES
    elif filter_name == "wan":
        layer_keys = WAN_LAYER_KEYNAMES
        all_avoid_keys = AVOID_KEY_NAMES
    else:
        # Custom comma-separated list
        layer_keys = []
        all_avoid_keys = AVOID_KEY_NAMES + [k.strip() for k in filter_name.split(",") if k.strip()]

    return all_avoid_keys, layer_keys


def generate_output_filename(
    input_file: str,
    target_dtype: torch.dtype,
    scaling_mode: str,
    filter_name: str,
    min_k: int = 1,
    max_k: int = 16,
    top_p: float = 0.01,
    lr: float = 1e-2,
) -> str:
    """
    Generate output filename for quantized model.

    Args:
        input_file: Input file path
        target_dtype: Target data type
        scaling_mode: Scaling mode
        filter_name: Filter name
        min_k: Minimum number of keys to keep
        max_k: Maximum number of keys to keep
        top_p: Top-p probability
        lr: Learning rate

    Returns:
        Output filename
    """
    base = os.path.splitext(input_file)[0]
    fp8_str = target_dtype.__str__().split('.')[-1]
    
    flags = f"_{filter_name}"
    
    # Format learning rate in scientific notation
    lr_str = f"{lr:.2e}"
    
    return (
        f"{base}_{fp8_str}_{scaling_mode}{flags}_"
        f"k{min_k}-{max_k}_p{top_p}_lr{lr_str}.safetensors"
    )
