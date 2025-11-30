"""
Utility functions for quantization operations.
"""

import torch
import gc
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


def should_process_layer(
    key: str,
    t5xxl: bool = False,
    keep_distillation_large: bool = False,
    keep_distillation_small: bool = False,
    keep_nerf_large: bool = False,
    keep_nerf_small: bool = False,
    radiance: bool = False,
    wan: bool = False,
    qwen: bool = False,
    hunyuan: bool = False,
    zimage_l: bool = False,
    zimage_s: bool = False,
) -> Tuple[bool, bool, str]:
    """
    Determine if a layer should be processed and if scale key should be created.
    
    Args:
        key: Layer key name
        t5xxl: T5XXL model flag
        keep_distillation_large: Keep large distillation layers
        keep_distillation_small: Keep small distillation layers
        keep_nerf_large: Keep large NeRF layers
        keep_nerf_small: Keep small NeRF layers
        radiance: Radiance field model flag
        wan: WAN model flag
        qwen: Qwen model flag
        hunyuan: Hunyuan model flag
        zimage_l: Z-Image large flag
        zimage_s: Z-Image small flag
        
    Returns:
        Tuple of (should_process, should_create_scale, skip_reason)
    """
    from convert_and_quantize.constants import (
        T5XXL_REMOVE_KEY_NAMES, AVOID_KEY_NAMES, RADIANCE_LAYER_KEYNAMES,
        WAN_LAYER_KEYNAMES, QWEN_AVOID_KEY_NAMES, QWEN_LAYER_KEYNAMES,
        ZIMAGE_AVOID_KEY_NAMES, ZIMAGE_LAYER_KEYNAMES, HUNYUAN_AVOID_KEY_NAMES,
        DISTILL_LAYER_KEYNAMES_LARGE, DISTILL_LAYER_KEYNAMES_SMALL,
        NERF_LAYER_KEYNAMES_LARGE, NERF_LAYER_KEYNAMES_SMALL
    )
    
    # Check removals
    if t5xxl and any(n in key for n in T5XXL_REMOVE_KEY_NAMES):
        return False, False, "T5XXL decoder/head removal"
    
    # Check exclusions
    if t5xxl and any(n in key for n in AVOID_KEY_NAMES):
        return False, False, "T5XXL exclusion"
    if radiance and any(n in key for n in RADIANCE_LAYER_KEYNAMES):
        return False, False, "Radiance exclusion"
    if wan and any(n in key for n in AVOID_KEY_NAMES):
        return False, False, "WAN exclusion"
    if qwen and any(n in key for n in QWEN_AVOID_KEY_NAMES):
        return False, False, "Qwen exclusion"
    if zimage_l and any(n in key for n in ZIMAGE_AVOID_KEY_NAMES):
        return False, False, "Z-Image exclusion"
    if zimage_s and any(n in key for n in ZIMAGE_AVOID_KEY_NAMES):
        return False, False, "Z-Image exclusion"
    if hunyuan and any(n in key for n in HUNYUAN_AVOID_KEY_NAMES):
        return False, False, "Hunyuan Video exclusion"
    
    # Check preservations
    if keep_distillation_large and any(n in key for n in DISTILL_LAYER_KEYNAMES_LARGE):
        return False, True, "Distillation layer (large)"
    if keep_distillation_small and any(n in key for n in DISTILL_LAYER_KEYNAMES_SMALL):
        return False, True, "Distillation layer (small)"
    if keep_nerf_large and any(n in key for n in NERF_LAYER_KEYNAMES_LARGE):
        return False, True, "NeRF layer (large)"
    if keep_nerf_small and any(n in key for n in NERF_LAYER_KEYNAMES_SMALL):
        return False, True, "NeRF layer (small)"
    if wan and any(n in key for n in WAN_LAYER_KEYNAMES):
        return False, True, "WAN layer preservation"
    if qwen and any(n in key for n in QWEN_LAYER_KEYNAMES):
        return False, True, "Qwen layer preservation"
    if zimage_l and any(n in key for n in ZIMAGE_LAYER_KEYNAMES):
        return False, True, "Z-Image layer preservation"
    
    return True, True, ""


def generate_output_filename(
    input_file: str,
    target_dtype: torch.dtype,
    scaling_mode: str,
    t5xxl: bool = False,
    keep_distillation_large: bool = False,
    keep_distillation_small: bool = False,
    keep_nerf_large: bool = False,
    keep_nerf_small: bool = False,
    radiance: bool = False,
    min_k: int = 1,
    max_k: int = 16,
    top_p: float = 0.01,
    lr: float = 1e-2,
) -> str:
    """
    Generate output filename based on configuration.
    
    Args:
        input_file: Input file path
        target_dtype: Target FP8 dtype
        scaling_mode: Scaling mode (tensor/block)
        Additional flags and parameters for naming
        
    Returns:
        Generated output filename
    """
    import os
    
    base = os.path.splitext(input_file)[0]
    fp8_str = target_dtype.__str__().split('.')[-1]
    
    flags = ""
    if t5xxl:
        flags += "_t5"
    if keep_distillation_large:
        flags += "_nodist_l"
    if keep_distillation_small:
        flags += "_nodist_s"
    if keep_nerf_large:
        flags += "_nonerf_l"
    if keep_nerf_small:
        flags += "_nonerf_s"
    if radiance:
        flags += "_norad"
    
    # Format learning rate in scientific notation
    lr_str = f"{lr:.2e}"
    
    return (
        f"{base}_{fp8_str}_{scaling_mode}{flags}_"
        f"k{min_k}-{max_k}_p{top_p}_lr{lr_str}.safetensors"
    )
