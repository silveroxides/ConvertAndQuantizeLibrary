"""
Convert and Quantize Library - Advanced quantization framework for neural networks.

This library provides tools for quantizing neural network models to FP8 format
with learned adaptive rounding and SVD-based optimization.
"""

__version__ = "0.1.0"
__author__ = "silveroxides"
__license__ = "MIT"

from convert_and_quantize.core import LearnedRoundingConverter
from convert_and_quantize.constants import (
    TARGET_FP8_DTYPE,
    COMPUTE_DTYPE,
    SCALE_DTYPE,
    FP8_MIN,
    FP8_MAX,
    FP8_MIN_POS,
)
from convert_and_quantize.utils import (
    get_device,
    get_fp8_constants,
    clean_gpu_memory,
    setup_seed,
    get_layer_filters,
    generate_output_filename,
)
from convert_and_quantize.optimizers import get_optimizer

__all__ = [
    "LearnedRoundingConverter",
    "TARGET_FP8_DTYPE",
    "COMPUTE_DTYPE",
    "SCALE_DTYPE",
    "FP8_MIN",
    "FP8_MAX",
    "FP8_MIN_POS",
    "get_device",
    "get_fp8_constants",
    "clean_gpu_memory",
    "setup_seed",
    "get_layer_filters",
    "generate_output_filename",
    "get_optimizer",
]
