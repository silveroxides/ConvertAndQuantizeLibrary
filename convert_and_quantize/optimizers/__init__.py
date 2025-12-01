"""
This module provides a factory function to get the requested optimizer.
"""

from .optimizer_impl import (
    optimize_with_original,
    optimize_with_adamw,
    optimize_with_radam,
    optimize_with_prodigy,
)


def get_optimizer(optimizer_name: str):
    """
    Returns the optimizer function based on the provided name.
    """
    optimizers = {
        "original": optimize_with_original,
        "adamw": optimize_with_adamw,
        "radam": optimize_with_radam,
        "ppsf": optimize_with_prodigy,
    }
    if optimizer_name not in optimizers:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")
    return optimizers[optimizer_name]
