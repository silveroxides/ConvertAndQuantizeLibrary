from __future__ import annotations
"""
Core quantization converter using learned adaptive rounding with SVD.
"""

import torch
import gc
import math
from typing import Tuple, Optional
from convert_and_quantize.constants import TARGET_FP8_DTYPE, COMPUTE_DTYPE, SCALE_DTYPE, T5XXL_REMOVE_KEY_NAMES
from convert_and_quantize.optimizers import get_optimizer
from convert_and_quantize.utils import setup_seed, MemoryEfficientSafeOpen


class LearnedRoundingConverter:
    """
    Implements advanced quantization using learned adaptive rounding.
    Provides a highly effective optimization strategy.
    """

    def __init__(
        self,
        optimizer: str = "original",
        num_iter: int = 500,
        top_p: float = 0.25,
        min_k: int = 64,
        max_k: int = 256,
        scaling_mode: str = 'tensor',
        block_size: int = 64,
        full_matrix: bool = False,
        seed: int = -1,
        lr: float = 1e-3,
        generator: Optional[torch.Generator] = None,
        device: Optional[torch.device] = None,
        **kwargs
    ):
        """
        Initialize the LearnedRoundingConverter.
        
        Args:
            optimizer: Optimizer choice ('original', 'adamw', 'radam', 'ppsf')
            num_iter: Number of optimization iterations
            top_p: Proportion of principal components to use
            min_k: Minimum number of principal components
            max_k: Maximum number of principal components
            scaling_mode: 'tensor' or 'block' scaling
            block_size: Block size for block scaling mode
            full_matrix: Use full SVD instead of lowrank
            seed: Random seed for reproducibility
            generator: Optional torch.Generator for reproducibility
            **kwargs: Additional optimizer-specific arguments
        """
        self.num_iter = num_iter
        self.top_p = top_p
        self.min_k = min_k
        self.max_k = max_k
        self.scaling_mode = scaling_mode
        self.block_size = block_size
        self.optimizer_choice = optimizer
        self.full_matrix = full_matrix
        self.seed = seed
        self.lr = lr
        self.generator = generator
        self.optimizer_kwargs = kwargs
        self.lr = lr
        self.f8_max_val = torch.finfo(TARGET_FP8_DTYPE).max

        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        print(f"LearnedRoundingConverter initialized on device: {self.device}")
        print(f"  - Using optimizer: '{self.optimizer_choice}'")
        print(f"  - Scaling mode: {self.scaling_mode}")
        if self.scaling_mode == 'block':
            print(f"    - Block size: {self.block_size}")

        self.optimizer = get_optimizer(self.optimizer_choice, lr=self.lr, **self.optimizer_kwargs)

    def convert(self, W_orig: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Convert a weight tensor to quantized FP8 format.
        
        Args:
            W_orig: Original weight tensor
            
        Returns:
            Tuple of (quantized_tensor, dequant_scale, dequantized_weight_tensor)
        """
        device = self.device
        setup_seed(self.seed, str(device), self.generator)
        W_float32 = W_orig.to(dtype=COMPUTE_DTYPE, device=device)

        # Handle zero tensors
        if torch.all(W_float32 == 0):
            print("  - Tensor is all zeros, skipping optimization.")
            quantized_tensor = torch.zeros_like(W_float32, dtype=TARGET_FP8_DTYPE)
            dequant_scale = None
            if self.scaling_mode == 'block' and W_float32.ndim == 2 and W_float32.shape[1] > 0 and W_float32.shape[1] % self.block_size == 0:
                out_features, in_features = W_float32.shape
                num_blocks = in_features // self.block_size
                dequant_scale = torch.ones(out_features, num_blocks, 1, device=self.device, dtype=SCALE_DTYPE)
            else:
                dequant_scale = torch.ones(1, device=self.device, dtype=SCALE_DTYPE)
            return quantized_tensor, dequant_scale, torch.zeros_like(W_float32)

        # Compute scaling factors
        scale = None
        compact_scale = None
        current_scaling_mode = self.scaling_mode

        if current_scaling_mode == 'block':
            if W_float32.ndim == 2 and W_float32.shape[1] > 0 and W_float32.shape[1] % self.block_size == 0:
                print(f"    - Using block scaling with block size {self.block_size}.")
                out_features, in_features = W_float32.shape
                num_blocks = in_features // self.block_size
                W_reshaped = W_float32.view(out_features, num_blocks, self.block_size)
                w_max = W_reshaped.abs().max(dim=2, keepdim=True)[0]
                compact_scale = self.f8_max_val / w_max.clamp_min_(1e-12)
                scale = compact_scale.repeat_interleave(self.block_size, dim=2).view(out_features, in_features)
            else:
                print(f"    - WARNING: Tensor shape {list(W_float32.shape)} not suitable for block size {self.block_size}. Falling back to 'tensor' scaling.")
                current_scaling_mode = 'tensor'

        if current_scaling_mode == 'tensor':
            w_max = W_float32.abs().max()
            scale = self.f8_max_val / w_max.clamp_min_(1e-12)
            compact_scale = scale

        # Compute SVD
        max_rank = min(W_float32.shape)
        k = min(self.max_k, max(self.min_k, int(math.floor(self.top_p * max_rank))))
        k = min(k, max_rank)

        print(f"    - Tensor shape: {list(W_float32.shape)}, Max rank: {max_rank}. Using k={k} components.")

        if self.full_matrix:
            print(f"Using torch.linalg.svd with full_matrices=True")
            U, _, Vh = torch.linalg.svd(W_float32, full_matrices=True, driver='gesvd')
        else:
            try:
                print(f"Trying svd_lowrank")
                U, _, Vh = torch.svd_lowrank(W_float32, q=min(k + 10, max_rank), niter=4)
                Vh = Vh.T
            except RuntimeError:
                print("    - svd_lowrank failed, falling back to full SVD.")
                U, _, Vh = torch.linalg.svd(W_float32, full_matrices=False)

        U_k, Vh_k = U[:, :k], Vh[:k, :]

        # Optimize
        assert scale is not None, "Scale must be initialized before optimization."
        final_tensor_scaled = self.optimizer(
            W_float32, scale, U_k, Vh_k,
            num_iter=self.num_iter,
            f8_max_val=self.f8_max_val,
            target_dtype=TARGET_FP8_DTYPE,
            compute_dtype=COMPUTE_DTYPE,
            **self.optimizer_kwargs
        )
        final_tensor_scaled.clamp_(-self.f8_max_val, self.f8_max_val)

        # Convert to FP8
        with torch.no_grad():
            W_f8 = final_tensor_scaled.to(dtype=TARGET_FP8_DTYPE)
            assert compact_scale is not None, "compact_scale should never be None at this point"
            assert scale is not None, "scale should never be None at this point"
            if current_scaling_mode == 'block':
                dequant_scale = compact_scale.reciprocal()
            else:
                dequant_scale = compact_scale.reciprocal().reshape(1)

            dequantized_weight_tensor = (W_f8.to(self.device, dtype=COMPUTE_DTYPE) / scale)

        del W_float32, scale, U, Vh, U_k, Vh_k, final_tensor_scaled, compact_scale
        gc.collect()
        if self.device == 'cuda':
            torch.cuda.empty_cache()

        return W_f8, dequant_scale.to(device=self.device, dtype=SCALE_DTYPE), dequantized_weight_tensor


def quantize_model(
    model_path: str,
    converter: LearnedRoundingConverter,
    calib_samples: int = 0,
    manual_seed: int = -1,
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
    mem_eff_safe_open: bool = False,
    seed_generator: Optional[torch.Generator] = None,
    device: Optional[torch.device] = None,
):
    from safetensors import safe_open
    from convert_and_quantize.constants import (
        AVOID_KEY_NAMES,
        T5XXL_REMOVE_KEY_NAMES,
        QWEN_AVOID_KEY_NAMES,
        HUNYUAN_AVOID_KEY_NAMES,
        ZIMAGE_AVOID_KEY_NAMES,
        RADIANCE_AVOID_KEY_NAMES,
        CHROMA_LAYER_KEYNAMES_LARGE,
        CHROMA_LAYER_KEYNAMES_SMALL,
        RADIANCE_LAYER_KEYNAMES_LARGE,
        RADIANCE_LAYER_KEYNAMES_SMALL,
        WAN_LAYER_KEYNAMES,
        QWEN_LAYER_KEYNAMES,
        ZIMAGE_LAYER_KEYNAMES,
    )

    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if not mem_eff_safe_open:
        open_fn = lambda fn: safe_open(fn, framework="pt", device='cpu')
    else:
        open_fn = lambda fn: MemoryEfficientSafeOpen(fn)
        print("Using memory-efficient safe_open for large models.")

    
    new_tensors = {}
    calibration_data_cache = {}
    print("Scanning model and generating simulated calibration data...")
    if calib_samples > 0:
        try:
            with open_fn(model_path) as f:
                for key in f.keys():
                    if key.endswith(".weight") and f.get_tensor(key).ndim == 2:
                        in_features = f.get_tensor(key).shape[1]
                        if in_features not in calibration_data_cache:
                            if seed_generator is None:
                                seed_generator = torch.Generator(device=device)
                                if manual_seed != -1:
                                    seed_generator.manual_seed(manual_seed)
                            
                            calibration_data_cache[in_features] = torch.randn(
                                calib_samples, 
                                in_features, 
                                dtype=COMPUTE_DTYPE, 
                                generator=seed_generator, 
                                device=device
                            )
        except FileNotFoundError:
            print(f"Model path not found: {model_path}")
            return {}
    print("Simulated calibration data generated.\n")

    with safe_open(model_path, framework="pt", device="cpu") as f:
        weight_keys = sorted([key for key in f.keys() if key.endswith('.weight') and f.get_tensor(key).ndim == 2])
        total_weights = len(weight_keys)
        print(f"Found {total_weights} weight tensors to potentially process.")
        
        for i, key in enumerate(weight_keys):
            process_this_key = True
            create_scale_key = True
            skip_reason = ""

            if t5xxl and any(n in key for n in T5XXL_REMOVE_KEY_NAMES):
                print(f"({i+1}/{total_weights}) Removing T5XXL decoder tensor: {key}")
                continue
            if t5xxl and any(n in key for n in AVOID_KEY_NAMES):
                skip_reason = "T5XXL exclusion"
                create_scale_key = False
                process_this_key = False
            if radiance and any(n in key for n in RADIANCE_AVOID_KEY_NAMES):
                skip_reason = "Radiance exclusion"
                create_scale_key = False
                process_this_key = False
            if wan and any(n in key for n in AVOID_KEY_NAMES):
                skip_reason = "WAN exclusion"
                create_scale_key = False
                process_this_key = False
            if qwen and any(n in key for n in QWEN_AVOID_KEY_NAMES):
                skip_reason = "Qwen Image exclusion"
                create_scale_key = False
                process_this_key = False
            if zimage_l and any(n in key for n in ZIMAGE_AVOID_KEY_NAMES):
                skip_reason = "Z-Image exclusion"
                create_scale_key = False
                process_this_key = False
            if zimage_s and any(n in key for n in ZIMAGE_AVOID_KEY_NAMES):
                skip_reason = "Z-Image exclusion"
                create_scale_key = False
                process_this_key = False
            if hunyuan and any(n in key for n in HUNYUAN_AVOID_KEY_NAMES):
                skip_reason = "Hunyuan Video 1.5 exclusion"
                create_scale_key = False
                process_this_key = False
            if chroma_large and any(n in key for n in CHROMA_LAYER_KEYNAMES_LARGE):
                skip_reason = "Distillation layer and Flux1 keep"
                create_scale_key = True
                process_this_key = False
            if chroma_small and any(n in key for n in CHROMA_LAYER_KEYNAMES_SMALL):
                skip_reason = "Distillation layer only"
                create_scale_key = True
                process_this_key = False
            if radiance_large and any(n in key for n in RADIANCE_LAYER_KEYNAMES_LARGE):
                skip_reason = "Distillation layer, NeRF layer and txt_in"
                create_scale_key = True
                process_this_key = False
            if radiance_small and any(n in key for n in RADIANCE_LAYER_KEYNAMES_SMALL):
                skip_reason = "Distillation layer and NeRF layer"
                create_scale_key = True
                process_this_key = False
            if wan and any(n in key for n in WAN_LAYER_KEYNAMES):
                skip_reason = "WAN layer keep in high"
                create_scale_key = True
                process_this_key = False
            if qwen and any(n in key for n in QWEN_LAYER_KEYNAMES):
                skip_reason = "Qwen Image layer keep in high"
                create_scale_key = True
                process_this_key = False
            if zimage_l and any(n in key for n in ZIMAGE_LAYER_KEYNAMES):
                skip_reason = "Z-Image layer keep in high"
                create_scale_key = True
                process_this_key = False

            tensor = f.get_tensor(key)
            if not process_this_key:
                if not create_scale_key:
                    print(f"({i+1}/{total_weights}) Skipping tensor: {key} (Reason: {skip_reason})")
                    new_tensors[key] = tensor.to(dtype=COMPUTE_DTYPE).to(device='cpu')
                else:
                    print(f"({i+1}/{total_weights}) Skipping tensor: {key} (Reason: {skip_reason})")
                    new_tensors[key] = tensor
                    base_name = key[:key.rfind('.weight')]
                    new_tensors[f"{base_name}.scale_weight"] = torch.tensor([1.0], dtype=SCALE_DTYPE).to(device='cpu')
                continue

            if tensor.ndim != 2:
                print(f"({i+1}/{total_weights}) Skipping tensor: {key} (Reason: Not a 2D tensor)")
                new_tensors[key] = tensor.to(dtype=COMPUTE_DTYPE).to(device='cpu')
                continue

            print(f"({i+1}/{total_weights}) Processing tensor: {key}")
            q_tensor, dequant_s, dequant_w = converter.convert(tensor)
            new_tensors[key] = q_tensor.to(device='cpu')
            base_name = key[:key.rfind('.weight')]
            new_tensors[f"{base_name}.scale_weight"] = dequant_s.to(device='cpu').detach().clone()
            
            if t5xxl:
                new_tensors[f"{base_name}.scale_input"] = dequant_s.to(device='cpu').detach().clone()

            bias_key = f"{base_name}.bias"
            if bias_key in f.keys():
                original_bias = f.get_tensor(bias_key)
                if calib_samples > 0 and tensor.shape[1] in calibration_data_cache:
                    X_calib = calibration_data_cache[tensor.shape[1]].to(device=device)
                    W_orig = tensor.to(device=device, dtype=COMPUTE_DTYPE)
                    W_dequant = dequant_w.to(device=device, dtype=COMPUTE_DTYPE)
                    
                    weight_error = W_orig - W_dequant
                    output_error = X_calib @ weight_error.T
                    bias_correction = output_error.mean(dim=0)
                    
                    new_bias = original_bias.to(device=device, dtype=COMPUTE_DTYPE) - bias_correction
                    new_tensors[bias_key] = new_bias.to(device='cpu', dtype=original_bias.dtype)
                    print(f"    - Original bias mean : {original_bias.mean().item():.6f}\n    - Corrected bias mean: {new_tensors[bias_key].mean().item():.6f}")
                else:
                    new_tensors[bias_key] = original_bias
                    if calib_samples > 0:
                        print(f"  - WARNING: No calibration data found for {bias_key}. Bias not corrected.")

        for key in f.keys():
            if key not in new_tensors:
                new_tensors[key] = f.get_tensor(key)

    if t5xxl:
        new_tensors["scaled_fp8"] = torch.empty((0), dtype=TARGET_FP8_DTYPE)
        
    return new_tensors