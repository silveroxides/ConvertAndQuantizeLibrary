"""
Optimizer implementations for quantization refinement.
"""

import torch
from tqdm import tqdm
from torch.optim import AdamW, RAdam
from typing import Callable

try:
    from prodigyplus.prodigy_plus_schedulefree import ProdigyPlusScheduleFree
    PRODIGY_AVAILABLE = True
except ImportError:
    PRODIGY_AVAILABLE = False


def optimize_with_original(
    W_float32: torch.Tensor,
    scale: torch.Tensor,
    U_k: torch.Tensor,
    Vh_k: torch.Tensor,
    num_iter: int = 500,
    lr: float = 0.5,
    f8_max_val: float = None,
    target_dtype: torch.dtype = None,
) -> torch.Tensor:
    """
    Original optimization algorithm with adaptive learning rate scheduling.
    """
    W_rounded = (W_float32 * scale).to(target_dtype).to(torch.float32)
    W_q_refined = W_rounded.clone()
    best_loss = float('inf')
    best_tensor = None
    worse_loss_counter = 0
    curr_lr = lr

    if W_float32.shape[0] == W_float32.shape[1]:
        small_mult = 0.95
    else:
        small_mult = 1.0

    pbar = tqdm(
        range(num_iter),
        desc="    Optimizing (Original)",
        leave=False,
        dynamic_ncols=True
    )
    
    for i in pbar:
        with torch.no_grad():
            current_dq = W_q_refined / scale
            error = current_dq - W_float32
            projected_error = U_k.T @ error @ Vh_k.T
            loss = torch.linalg.norm(projected_error)

        if loss.item() < best_loss and worse_loss_counter < 50:
            best_loss = loss.item()
            best_tensor = W_q_refined.clone()
            worse_loss_counter = 0
            curr_lr = min(curr_lr * (1.20 * small_mult), 100.0)
        elif loss.item() < best_loss and worse_loss_counter > 49 and worse_loss_counter < 75:
            best_loss = loss.item()
            best_tensor = W_q_refined.clone()
            worse_loss_counter = 35
            curr_lr = min(curr_lr * (1.40 * small_mult), 100.0)
        elif loss.item() < best_loss and worse_loss_counter > 74 and worse_loss_counter < 100:
            best_loss = loss.item()
            best_tensor = W_q_refined.clone()
            worse_loss_counter = 65
            curr_lr = min(curr_lr * (1.60 * small_mult), 100.0)
        elif loss.item() < best_loss and worse_loss_counter > 99 and worse_loss_counter < 150:
            best_loss = loss.item()
            best_tensor = W_q_refined.clone()
            worse_loss_counter = 96
            curr_lr = min(curr_lr * (1.80 * small_mult), 100.0)
        elif loss.item() < best_loss and worse_loss_counter > 149 and worse_loss_counter < 200:
            best_loss = loss.item()
            best_tensor = W_q_refined.clone()
            worse_loss_counter = 147
            curr_lr = min(curr_lr * (2.0 * small_mult), 100.0)
        elif loss.item() < best_loss and worse_loss_counter > 199 and worse_loss_counter < 300:
            best_loss = loss.item()
            best_tensor = W_q_refined.clone()
            worse_loss_counter = 198
            curr_lr = min(curr_lr * (2.25 * small_mult), 100.0)
        elif loss.item() < best_loss and worse_loss_counter > 299 and worse_loss_counter < 400:
            best_loss = loss.item()
            best_tensor = W_q_refined.clone()
            worse_loss_counter = 299
            curr_lr = min(curr_lr * (2.5 * small_mult), 100.0)
        elif loss.item() < best_loss and worse_loss_counter > 399 and worse_loss_counter < 500:
            best_loss = loss.item()
            best_tensor = W_q_refined.clone()
            worse_loss_counter = 400
            curr_lr = min(curr_lr * (2.75 * small_mult), 100.0)
        elif loss.item() < best_loss and worse_loss_counter > 499:
            best_loss = loss.item()
            best_tensor = W_q_refined.clone()
            worse_loss_counter = 500
            curr_lr = min(curr_lr * (3.0 * small_mult), 100.0)
        elif loss.item() > best_loss and worse_loss_counter < 26:
            worse_loss_counter += 1
            curr_lr = max(curr_lr * (0.95 * small_mult), 1e-8)
        elif worse_loss_counter > 25 and worse_loss_counter < 76:
            worse_loss_counter += 1
            curr_lr = max(curr_lr * (0.975 * small_mult), 1e-8)
        elif worse_loss_counter > 75 and worse_loss_counter < 151:
            worse_loss_counter += 1
            curr_lr = max(curr_lr * (0.9875 * small_mult), 1e-8)
        elif worse_loss_counter > 150 and worse_loss_counter < 201:
            worse_loss_counter += 1
            curr_lr = max(curr_lr * (0.99 * small_mult), 1e-8)
        elif worse_loss_counter > 200 and worse_loss_counter < 301:
            worse_loss_counter += 1
            curr_lr = max(curr_lr * (0.995 * small_mult), 1e-8)
        elif worse_loss_counter > 300 and worse_loss_counter < 401:
            worse_loss_counter += 1
            curr_lr = max(curr_lr * (0.9975 * small_mult), 1e-8)
        elif worse_loss_counter > 400 and worse_loss_counter < 501:
            worse_loss_counter += 1
            curr_lr = max(curr_lr * (0.99875 * small_mult), 1e-8)
        elif worse_loss_counter > 500 and worse_loss_counter < 601:
            worse_loss_counter += 1
            curr_lr = max(curr_lr * (0.999 * small_mult), 1e-8)
        elif worse_loss_counter > 600:
            worse_loss_counter += 1
            curr_lr = max(curr_lr * (0.9995 * small_mult), 1e-8)

        if worse_loss_counter > 1000 and curr_lr > 1e-6:
            print("      - Learning Rate not converging. Resetting worse_loss_counter to 0.")
            worse_loss_counter = 0

        pbar.set_postfix({
            "loss": f"{loss.item():.3e}",
            "best": f"{best_loss:.3e}",
            "lr": f"{curr_lr:.2e}",
            "worse_count": f"{worse_loss_counter}"
        })

        if loss.item() < 1e-9 or curr_lr < 2e-8 or worse_loss_counter > 1000:
            if worse_loss_counter > 1000:
                print("      - Loss has stalled. Stopping.")
            elif loss.item() < 1e-9:
                print("      - Loss is too small. Stopping.")
            elif curr_lr < 2e-8:
                print("      - Learning Rate is too small. Stopping.")
            else:
                print("      - Loss is negligible. Stopping.")
            break

        with torch.no_grad():
            grad_direction = U_k @ (projected_error / loss.clamp_min(1e-20)) @ Vh_k
            W_q_refined -= curr_lr * (grad_direction * scale)

    pbar.close()
    return best_tensor if best_tensor is not None else W_q_refined


def optimize_with_adamw(
    W_float32: torch.Tensor,
    scale: torch.Tensor,
    U_k: torch.Tensor,
    Vh_k: torch.Tensor,
    num_iter: int = 500,
    lr: float = 1e-2,
    f8_max_val: float = None,
    target_dtype: torch.dtype = None,
) -> torch.Tensor:
    """
    AdamW optimizer for quantization refinement.
    """
    W_rounded = (W_float32 * scale).to(target_dtype).to(torch.float32)
    delta = torch.zeros_like(W_rounded, requires_grad=True)
    optimizer = AdamW([delta], lr=lr)
    best_loss = float('inf')
    best_delta = delta.detach().clone()

    pbar = tqdm(
        range(num_iter),
        desc="    Optimizing (AdamW)",
        leave=False,
        dynamic_ncols=True
    )
    
    for i in pbar:
        optimizer.zero_grad()
        W_q_refined = W_rounded + delta

        current_dq = W_q_refined / scale
        error = current_dq - W_float32
        projected_error = U_k.T @ error @ Vh_k.T
        loss = torch.linalg.norm(projected_error)

        loss.backward()
        optimizer.step()

        current_loss_val = loss.item()
        if current_loss_val < best_loss:
            best_loss = current_loss_val
            best_delta = delta.detach().clone()

        pbar.set_postfix({"loss": f"{current_loss_val:.3e}", "best": f"{best_loss:.3e}"})
        
        if best_loss < 1e-8:
            print("      - Loss is negligible. Stopping early.")
            break

    pbar.close()
    return W_rounded + best_delta


def optimize_with_radam(
    W_float32: torch.Tensor,
    scale: torch.Tensor,
    U_k: torch.Tensor,
    Vh_k: torch.Tensor,
    num_iter: int = 500,
    lr: float = 1e-2,
    f8_max_val: float = None,
    target_dtype: torch.dtype = None,
) -> torch.Tensor:
    """
    RAdam optimizer for quantization refinement.
    """
    W_rounded = (W_float32 * scale).to(target_dtype).to(torch.float32)
    delta = torch.zeros_like(W_rounded, requires_grad=True)
    optimizer = RAdam([delta], lr=lr)
    best_loss = float('inf')
    best_delta = delta.detach().clone()

    pbar = tqdm(
        range(num_iter),
        desc="    Optimizing (RAdam)",
        leave=False,
        dynamic_ncols=True
    )
    
    for i in pbar:
        optimizer.zero_grad()
        W_q_refined = W_rounded + delta

        current_dq = W_q_refined / scale
        error = current_dq - W_float32
        projected_error = U_k.T @ error @ Vh_k.T
        loss = torch.linalg.norm(projected_error)

        loss.backward()
        optimizer.step()

        current_loss_val = loss.item()
        if current_loss_val < best_loss:
            best_loss = current_loss_val
            best_delta = delta.detach().clone()

        pbar.set_postfix({"loss": f"{current_loss_val:.3e}", "best": f"{best_loss:.3e}"})
        
        if best_loss < 1e-8:
            print("      - Loss is negligible. Stopping early.")
            break

    pbar.close()
    return W_rounded + best_delta


def optimize_with_prodigy(
    W_float32: torch.Tensor,
    scale: torch.Tensor,
    U_k: torch.Tensor,
    Vh_k: torch.Tensor,
    num_iter: int = 500,
    lr: float = 1e-2,
    f8_max_val: float = None,
    target_dtype: torch.dtype = None,
) -> torch.Tensor:
    """
    ProdigyPlusScheduleFree optimizer for quantization refinement.
    """
    if not PRODIGY_AVAILABLE:
        raise ImportError(
            "ProdigyPlusScheduleFree not available. "
            "Install with: pip install prodigyplus"
        )

    W_rounded = (W_float32 * scale).to(target_dtype).to(torch.float32)
    delta = torch.zeros_like(W_rounded, requires_grad=True)
    optimizer = ProdigyPlusScheduleFree(
        [delta], lr=lr, betas=(0.9, 0.99), beta3=None,
        weight_decay=0.0, weight_decay_by_lr=False, d0=1e-3, d_coef=1.0,
        d_limiter=True, prodigy_steps=0, schedulefree_c=0, eps=1e-8,
        split_groups=False, split_groups_mean=False,
        factored=True, factored_fp32=True, use_bias_correction=False,
        use_stableadamw=True, use_schedulefree=True, use_speed=False,
        stochastic_rounding=True, fused_back_pass=False,
        use_cautious=False, use_grams=False, use_adopt=False,
        use_orthograd=False, use_focus=False
    )
    best_loss = float('inf')
    best_delta = delta.detach().clone()

    pbar = tqdm(
        range(num_iter),
        desc="    Optimizing (ProdigyPlusScheduleFree)",
        leave=False,
        dynamic_ncols=True
    )
    
    for i in pbar:
        optimizer.zero_grad()
        W_q_refined = W_rounded + delta

        current_dq = W_q_refined / scale
        error = current_dq - W_float32
        projected_error = U_k.T @ error @ Vh_k.T
        loss = torch.linalg.norm(projected_error)

        loss.backward()
        optimizer.step()

        current_loss_val = loss.item()
        if current_loss_val < best_loss:
            best_loss = current_loss_val
            best_delta = delta.detach().clone()

        pbar.set_postfix({"loss": f"{current_loss_val:.3e}", "best": f"{best_loss:.3e}"})
        
        if best_loss < 1e-8:
            print("      - Loss is negligible. Stopping early.")
            break

    pbar.close()
    return W_rounded + best_delta


# Optimizer registry
OPTIMIZERS = {
    "original": optimize_with_original,
    "adamw": optimize_with_adamw,
    "radam": optimize_with_radam,
    "ppsf": optimize_with_prodigy,
}


def get_optimizer(name: str) -> Callable:
    """
    Get optimizer function by name.
    
    Args:
        name: Optimizer name ('original', 'adamw', 'radam', 'ppsf')
        
    Returns:
        Optimizer function
        
    Raises:
        ValueError: If optimizer name is not recognized
    """
    if name not in OPTIMIZERS:
        raise ValueError(
            f"Unknown optimizer: '{name}'. "
            f"Available: {list(OPTIMIZERS.keys())}"
        )
    return OPTIMIZERS[name]
