"""
This module provides a factory function to get the requested optimizer.
"""

from .optimizer_impl import (
    optimize_with_original,
    optimize_with_adamw,
    optimize_with_radam,
    optimize_with_prodigy,
)


def get_optimizer(optimizer_name: str, lr: float = 1e-2, **kwargs):
    """
    Returns the optimizer function based on the provided name.
    """
    optimizers = {
        "original": lambda W_float32, scale, U_k, Vh_k, num_iter, target_dtype, compute_dtype, **opt_kwargs: optimize_with_original(W_float32, scale, U_k, Vh_k, num_iter, lr, target_dtype, compute_dtype, **opt_kwargs),
        "adamw": lambda W_float32, scale, U_k, Vh_k, num_iter, target_dtype, compute_dtype, **opt_kwargs: optimize_with_adamw(W_float32, scale, U_k, Vh_k, num_iter, lr, target_dtype, compute_dtype, **opt_kwargs),
        "radam": lambda W_float32, scale, U_k, Vh_k, num_iter, target_dtype, compute_dtype, **opt_kwargs: optimize_with_radam(W_float32, scale, U_k, Vh_k, num_iter, lr, target_dtype, compute_dtype, **opt_kwargs),
        "ppsf": lambda W_float32, scale, U_k, Vh_k, num_iter, target_dtype, compute_dtype, **opt_kwargs: optimize_with_prodigy(W_float32, scale, U_k, Vh_k, num_iter, lr, target_dtype, compute_dtype, **opt_kwargs),
    }
    if optimizer_name not in optimizers:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")
    return optimizers[optimizer_name]
