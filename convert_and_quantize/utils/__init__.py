"""
Utility functions for quantization operations.
"""

import torch
import gc
import os
from typing import Tuple, Optional
import json
import struct



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


def setup_seed(seed: int, device: Optional[str] = None, generator: Optional[torch.Generator] = None) -> torch.Generator:
    """
    Set up reproducible random seed.
    
    Args:
        seed: Random seed value
        device: Device for generator
        generator: Optional existing generator
        
    Returns:
        Generator object
    """
    if generator is None:
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
        CHROMA_LAYER_KEYNAMES_LARGE, CHROMA_LAYER_KEYNAMES_SMALL,
        RADIANCE_LAYER_KEYNAMES_LARGE, RADIANCE_LAYER_KEYNAMES_SMALL,
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
        layer_keys = CHROMA_LAYER_KEYNAMES_LARGE
        all_avoid_keys = AVOID_KEY_NAMES
    elif filter_name == "chroma_s":
        layer_keys = CHROMA_LAYER_KEYNAMES_SMALL
        all_avoid_keys = AVOID_KEY_NAMES
    elif filter_name == "nerf_l":
        layer_keys = RADIANCE_LAYER_KEYNAMES_LARGE
        all_avoid_keys = AVOID_KEY_NAMES + RADIANCE_AVOID_KEY_NAMES
    elif filter_name == "nerf_s":
        layer_keys = RADIANCE_LAYER_KEYNAMES_SMALL
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


def should_process_layer(
    key: str,
    t5xxl: bool = False,
    chroma_large: bool = False,
    chroma_small: bool = False,
    radiance_large: bool = False,
    radiance_small: bool = False,
    radiance: bool = False,
    wan: bool = False,
    qwen: bool = False,
    hunyuan: bool = False,
    zimage_l: bool = False,
    zimage_s: bool = False,
) -> tuple:
    """
    Decide whether to process (quantize) a layer.

    Returns:
        (should_process: bool, create_scale: bool, reason: str)
    """
    from convert_and_quantize.constants import (
        AVOID_KEY_NAMES,
        ZIMAGE_AVOID_KEY_NAMES,
        ZIMAGE_LAYER_KEYNAMES,
        QWEN_AVOID_KEY_NAMES,
        QWEN_LAYER_KEYNAMES,
        HUNYUAN_AVOID_KEY_NAMES,
        CHROMA_LAYER_KEYNAMES_LARGE,
        CHROMA_LAYER_KEYNAMES_SMALL,
        RADIANCE_LAYER_KEYNAMES_LARGE,
        RADIANCE_LAYER_KEYNAMES_SMALL,
        RADIANCE_AVOID_KEY_NAMES,
        WAN_LAYER_KEYNAMES,
        T5XXL_REMOVE_KEY_NAMES,
    )

    # Build avoid and high-precision lists based on flags
    all_avoid = list(AVOID_KEY_NAMES)
    layer_keys = []

    if zimage_l:
        layer_keys = ZIMAGE_LAYER_KEYNAMES
        all_avoid = ZIMAGE_AVOID_KEY_NAMES + all_avoid
    elif zimage_s:
        layer_keys = ZIMAGE_LAYER_KEYNAMES
        all_avoid = ZIMAGE_AVOID_KEY_NAMES + all_avoid
    if qwen:
        layer_keys = QWEN_LAYER_KEYNAMES
        all_avoid = QWEN_AVOID_KEY_NAMES + all_avoid
    if hunyuan:
        all_avoid = HUNYUAN_AVOID_KEY_NAMES + all_avoid
    if wan:
        layer_keys = WAN_LAYER_KEYNAMES
        all_avoid = AVOID_KEY_NAMES + all_avoid
    if chroma_large:
        layer_keys = CHROMA_LAYER_KEYNAMES_LARGE
        all_avoid = AVOID_KEY_NAMES + all_avoid
    if chroma_small:
        layer_keys = CHROMA_LAYER_KEYNAMES_SMALL
        all_avoid = AVOID_KEY_NAMES + all_avoid
    if radiance_large:
        layer_keys = RADIANCE_LAYER_KEYNAMES_LARGE
        all_avoid = AVOID_KEY_NAMES + RADIANCE_AVOID_KEY_NAMES + all_avoid
    if radiance_small:
        layer_keys = RADIANCE_LAYER_KEYNAMES_SMALL
        all_avoid = AVOID_KEY_NAMES + RADIANCE_AVOID_KEY_NAMES + all_avoid
    if radiance:
        all_avoid = AVOID_KEY_NAMES + RADIANCE_AVOID_KEY_NAMES + all_avoid
    if t5xxl:
        # T5-XXL has some decoder tensors that are removed entirely
        if any(n in key for n in T5XXL_REMOVE_KEY_NAMES):
            return False, False, "T5XXL remove list"

    # Check avoid list
    if any(n in key for n in all_avoid):
        return False, False, "In avoid list"

    # Check high-precision list
    if any(n in key for n in layer_keys):
        return False, True, "In high-precision list"

    return True, True, ""

class MemoryEfficientSafeOpen:
    def __init__(self, filename, device='cpu'):
        self.filename = filename
        self.device = device
        self.header, self.header_size = self._read_header()
        self.file = open(filename, "rb")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.file.close()

    def keys(self):
        return [k for k in self.header.keys() if k != "__metadata__"]

    def get_tensor(self, key):
        if key not in self.header:
            raise KeyError(f"Tensor '{key}' not found in the file")

        metadata = self.header[key]
        offset_start, offset_end = metadata["data_offsets"]

        tensor_bytes = None
        if offset_start != offset_end:
            self.file.seek(self.header_size + 8 + offset_start)
            tensor_bytes = self.file.read(offset_end - offset_start)

        return self._deserialize_tensor(tensor_bytes, metadata)

    def _read_header(self):
        with open(self.filename, "rb") as f:
            header_size = struct.unpack("<Q", f.read(8))[0]
            header_json = f.read(header_size).decode("utf-8")
        return json.loads(header_json), header_size

    def _deserialize_tensor(self, tensor_bytes, metadata):
        dtype_str = metadata["dtype"]
        shape = metadata["shape"]
        dtype = self._get_torch_dtype(dtype_str)

        if tensor_bytes is None:
            byte_tensor = torch.empty(0, dtype=torch.uint8)
        else:
            byte_tensor = torch.frombuffer(bytearray(tensor_bytes), dtype=torch.uint8)

        if dtype_str in ["F8_E5M2", "F8_E4M3"]:
            return self._convert_float8(byte_tensor, dtype_str, shape)

        return byte_tensor.view(dtype).reshape(shape)

    @staticmethod
    def _get_torch_dtype(dtype_str):
        dtype_map = {
            "F64": torch.float64, "F32": torch.float32, "F16": torch.float16, "BF16": torch.bfloat16,
            "I64": torch.int64, "I32": torch.int32, "I16": torch.int16, "I8": torch.int8,
            "U8": torch.uint8, "BOOL": torch.bool,
        }
        if hasattr(torch, "float8_e5m2"):
            dtype_map["F8_E5M2"] = torch.float8_e5m2
        if hasattr(torch, "float8_e4m3fn"):
            dtype_map["F8_E4M3"] = torch.float8_e4m3fn

        dtype = dtype_map.get(dtype_str)
        if dtype is None:
            raise ValueError(f"Unsupported dtype: {dtype_str}")
        return dtype

    @staticmethod
    def _convert_float8(byte_tensor, dtype_str, shape):
        if dtype_str == "F8_E5M2" and hasattr(torch, "float8_e5m2"):
            return byte_tensor.view(torch.float8_e5m2).reshape(shape)
        elif dtype_str == "F8_E4M3" and hasattr(torch, "float8_e4m3fn"):
            return byte_tensor.view(torch.float8_e4m3fn).reshape(shape)
        else:
            raise ValueError(f"Unsupported float8 type: {dtype_str}. Your PyTorch version may be too old.")
