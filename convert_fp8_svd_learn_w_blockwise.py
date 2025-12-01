import argparse
import os
import torch
from safetensors import safe_open
from safetensors.torch import save_file
from typing import Dict, Tuple
from tqdm import tqdm
import gc
import math
from torch.optim import AdamW, RAdam
from prodigyplus.prodigy_plus_schedulefree import ProdigyPlusScheduleFree

# --- Constants and Configuration ---
AVOID_KEY_NAMES = ["norm", "bias", "embed_tokens", "shared", "patch_embedding", "audio_model.patch_embedding", "ref_conv", "control_adapter", "motion_encoder.enc.net_app", "face_encoder.conv", "pose_patch_embedding", "motion_encoder.enc.fc", "img_emb.proj", "q_norm", "motion_encoder.dec", "head.modulation", "casual_audio_encoder", "cond_encoder", "frame_packer", "norm_k", "norm_q"]
T5XXL_REMOVE_KEY_NAMES = ["decoder", "lm_head"]
QWEN_AVOID_KEY_NAMES = ["norm_added_k", "norm_added_q", "norm_k", "norm_q", "txt_norm"]
HUNYUAN_AVOID_KEY_NAMES = ["layernorm", "img_attn_k_norm", "img_attn_q_norm", "txt_attn_k_norm", "txt_attn_q_norm", "norm1", "norm2", "vision_in.proj.0", "vision_in.proj.4", "img_in.proj", "cond_type_embedding"]
ZIMAGE_AVOID_KEY_NAMES = ["cap_embedder.0", "cap_pad_token", "attention_norm1", "attention_norm2", "ffn_norm1", "ffn_norm2", "k_norm", "q_norm", "x_pad_token"]
DISTILL_LAYER_KEYNAMES_LARGE = ["distilled_guidance_layer", "final_layer", "img_in", "txt_in"]
DISTILL_LAYER_KEYNAMES_SMALL = ["distilled_guidance_layer"]
NERF_LAYER_KEYNAMES_LARGE = ["distilled_guidance_layer", "nerf_blocks", "nerf_image_embedder", "txt_in"]
NERF_LAYER_KEYNAMES_SMALL = ["distilled_guidance_layer", "nerf_blocks", "nerf_image_embedder"]
RADIANCE_LAYER_KEYNAMES = ["img_in_patch", "nerf_final_layer_conv"]
WAN_LAYER_KEYNAMES = ["text_embedding", "time_embedding", "audio_model.text_embedding", "audio_model.time_embedding", "time_projection", "video_model.time_projection", "head.head", "face_encoder.out_proj", "face_adapter"]
QWEN_LAYER_KEYNAMES = ["time_text_embed", "img_in", "norm_out", "proj_out", "txt_in"]
ZIMAGE_LAYER_KEYNAMES = ["x_embedder", "final_layer", "cap_embedder.1", "adaLN_modulation", "t_embedder"]
TARGET_FP8_DTYPE = torch.float8_e4m3fn
COMPUTE_DTYPE = torch.float32
SCALE_DTYPE = torch.float32

class LearnedRoundingConverter:
    """
    Implements advanced quantization using learned adaptive rounding.
    Provides a highly effective optimization strategy.
    """
    def __init__(self, optimizer="original", num_iter=500, top_p=0.01, min_k=1, max_k=16, scaling_mode='tensor', block_size=64, full_matrix=False, **kwargs):
        self.num_iter = num_iter
        self.top_p = top_p
        self.min_k = min_k
        self.max_k = max_k
        self.scaling_mode = scaling_mode
        self.block_size = block_size
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.optimizer_choice = optimizer
        self.full_matrix = full_matrix
        self.optimizer_kwargs = kwargs
        self.f8_max_val = torch.finfo(TARGET_FP8_DTYPE).max

        print(f"LearnedRoundingConverter initialized on device: {self.device}")
        print(f"  - Using optimizer: '{self.optimizer_choice}'")
        print(f"  - Scaling mode: {self.scaling_mode}")
        if self.scaling_mode == 'block':
            print(f"    - Block size: {self.block_size}")

    def _optimize_ppsf(self, W_float32: torch.Tensor, scale: torch.Tensor, U_k: torch.Tensor, Vh_k: torch.Tensor) -> torch.Tensor:
        W_rounded = (W_float32 * scale).to(TARGET_FP8_DTYPE).to(COMPUTE_DTYPE)
        delta = torch.zeros_like(W_rounded, requires_grad=True)
        lr = self.optimizer_kwargs.get('lr', 1e-2)
        optimizer = ProdigyPlusScheduleFree([delta], lr=lr, betas=(0.9, 0.99), beta3=None, 
                                            weight_decay=0.0, weight_decay_by_lr=False, d0=1e-3, d_coef=1.0,
                                            d_limiter=True, prodigy_steps=0, schedulefree_c=0, eps=1e-8,
                                            split_groups=False, split_groups_mean=False,
                                            factored=True, factored_fp32=True, use_bias_correction=False,
                                            use_stableadamw=True, use_schedulefree=True, use_speed=False,
                                            stochastic_rounding=True, fused_back_pass=False,
                                            use_cautious=False, use_grams=False, use_adopt=False,
                                            use_orthograd=False, use_focus=False)
        best_loss = float('inf')
        best_delta = delta.detach().clone()

        pbar = tqdm(range(self.num_iter), desc="    Optimizing (ProdigyPlusScheduleFree)", leave=False, dynamic_ncols=True)
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
                print(f"      - Loss is negligible. Stopping early.")
                break

        pbar.close()
        return W_rounded + best_delta

    def _optimize_adamw(self, W_float32: torch.Tensor, scale: torch.Tensor, U_k: torch.Tensor, Vh_k: torch.Tensor) -> torch.Tensor:
        W_rounded = (W_float32 * scale).to(TARGET_FP8_DTYPE).to(COMPUTE_DTYPE)
        delta = torch.zeros_like(W_rounded, requires_grad=True)
        lr = self.optimizer_kwargs.get('lr', 1e-2)
        optimizer = AdamW([delta], lr=lr)
        best_loss = float('inf')
        best_delta = delta.detach().clone()

        pbar = tqdm(range(self.num_iter), desc="    Optimizing (AdamW)", leave=False, dynamic_ncols=True)
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
                print(f"      - Loss is negligible. Stopping early.")
                break

        pbar.close()
        return W_rounded + best_delta

    def _optimize_radam(self, W_float32: torch.Tensor, scale: torch.Tensor, U_k: torch.Tensor, Vh_k: torch.Tensor) -> torch.Tensor:
        W_rounded = (W_float32 * scale).to(TARGET_FP8_DTYPE).to(COMPUTE_DTYPE)
        delta = torch.zeros_like(W_rounded, requires_grad=True)
        lr = self.optimizer_kwargs.get('lr', 1e-2)
        optimizer = RAdam([delta], lr=lr)
        best_loss = float('inf')
        best_delta = delta.detach().clone()

        pbar = tqdm(range(self.num_iter), desc="    Optimizing (RAdam)", leave=False, dynamic_ncols=True)
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
                print(f"      - Loss is negligible. Stopping early.")
                break

        pbar.close()
        return W_rounded + best_delta

    def _optimize_original(self, W_float32: torch.Tensor, scale: torch.Tensor, U_k: torch.Tensor, Vh_k: torch.Tensor) -> torch.Tensor:
        W_rounded = (W_float32 * scale).to(TARGET_FP8_DTYPE).to(COMPUTE_DTYPE)
        W_q_refined = W_rounded.clone()
        best_loss = float('inf')
        best_tensor = None
        worse_loss_counter = 0
        curr_lr = self.optimizer_kwargs.get('lr', 0.5)
        if W_float32.shape[0] == W_float32.shape[1]:
            small_mult = 0.95
        else:
            small_mult = 1.0

        pbar = tqdm(range(self.num_iter), desc="    Optimizing (Original)", leave=False, dynamic_ncols=True)
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
                curr_lr = min(curr_lr * (1.25 * small_mult), 100.0)
            elif loss.item() < best_loss and worse_loss_counter > 49 and worse_loss_counter < 75:
                best_loss = loss.item()
                best_tensor = W_q_refined.clone()
                worse_loss_counter = 35
                curr_lr = min(curr_lr * (1.5 * small_mult), 100.0)
            elif loss.item() < best_loss and worse_loss_counter > 74 and worse_loss_counter < 100:
                best_loss = loss.item()
                best_tensor = W_q_refined.clone()
                worse_loss_counter = 65
                curr_lr = min(curr_lr * (1.75 * small_mult), 100.0)
            elif loss.item() < best_loss and worse_loss_counter > 99 and worse_loss_counter < 150:
                best_loss = loss.item()
                best_tensor = W_q_refined.clone()
                worse_loss_counter = 96
                curr_lr = min(curr_lr * (2.0 * small_mult), 100.0)
            elif loss.item() < best_loss and worse_loss_counter > 149 and worse_loss_counter < 200:
                best_loss = loss.item()
                best_tensor = W_q_refined.clone()
                worse_loss_counter = 147
                curr_lr = min(curr_lr * (2.25 * small_mult), 100.0)
            elif loss.item() < best_loss and worse_loss_counter > 199 and worse_loss_counter < 300:
                best_loss = loss.item()
                best_tensor = W_q_refined.clone()
                worse_loss_counter = 198
                curr_lr = min(curr_lr * (2.5 * small_mult), 100.0)
            elif loss.item() < best_loss and worse_loss_counter > 299 and worse_loss_counter < 400:
                best_loss = loss.item()
                best_tensor = W_q_refined.clone()
                worse_loss_counter = 299
                curr_lr = min(curr_lr * (2.75 * small_mult), 100.0)
            elif loss.item() < best_loss and worse_loss_counter > 399 and worse_loss_counter < 500:
                best_loss = loss.item()
                best_tensor = W_q_refined.clone()
                worse_loss_counter = 400
                curr_lr = min(curr_lr * (3.0 * small_mult), 100.0)
            elif loss.item() < best_loss and worse_loss_counter > 499:
                best_loss = loss.item()
                best_tensor = W_q_refined.clone()
                worse_loss_counter = 500
                curr_lr = min(curr_lr * (3.25 * small_mult), 100.0)
            elif loss.item() > best_loss and worse_loss_counter < 26:
                worse_loss_counter += 1
                curr_lr = max(curr_lr * (0.95 * small_mult), 1e-8)
            elif worse_loss_counter > 25 and worse_loss_counter < 76:
                worse_loss_counter += 1
                curr_lr = max(curr_lr * (0.9625 * small_mult), 1e-8)
            elif worse_loss_counter > 75 and worse_loss_counter < 151:
                worse_loss_counter += 1
                curr_lr = max(curr_lr * (0.975 * small_mult), 1e-8)
            elif worse_loss_counter > 150 and worse_loss_counter < 201:
                worse_loss_counter += 1
                curr_lr = max(curr_lr * (0.9875 * small_mult), 1e-8)
            elif worse_loss_counter > 200 and worse_loss_counter < 301:
                worse_loss_counter += 1
                curr_lr = max(curr_lr * (0.99 * small_mult), 1e-8)
            elif worse_loss_counter > 300 and worse_loss_counter < 401:
                worse_loss_counter += 1
                curr_lr = max(curr_lr * (0.99125 * small_mult), 1e-8)
            elif worse_loss_counter > 400 and worse_loss_counter < 501:
                worse_loss_counter += 1
                curr_lr = max(curr_lr * (0.9925 * small_mult), 1e-8)
            elif worse_loss_counter > 500 and worse_loss_counter < 601:
                worse_loss_counter += 1
                curr_lr = max(curr_lr * (0.99375 * small_mult), 1e-8)
            elif worse_loss_counter > 600:
                worse_loss_counter += 1
                curr_lr = max(curr_lr * (0.995 * small_mult), 1e-8)


            pbar.set_postfix({"loss": f"{loss.item():.3e}", "best": f"{best_loss:.3e}", "lr": f"{curr_lr:.2e}", "worse_count": f"{worse_loss_counter}"})

            if loss.item() < 1e-9 or curr_lr < 2e-08 or worse_loss_counter > 1500:
                if worse_loss_counter > 1500:
                    print("      - Loss has stalled. Stopping.")
                elif curr_lr < 2e-8:
                    print("      - Learning Rate has bottomed out. Stopping.")
                else:
                    print("      - Loss is negligible. Stopping.")
                break

            with torch.no_grad():
                grad_direction = U_k @ (projected_error / loss.clamp_min(1e-20)) @ Vh_k
                W_q_refined -= curr_lr * (grad_direction * scale)

        pbar.close()
        return best_tensor if best_tensor is not None else W_q_refined

    def convert(self, W_orig: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        W_float32 = W_orig.to(self.device, dtype=COMPUTE_DTYPE)

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
        
        assert scale is not None, "scale should not be None after scaling mode selection"

        max_rank = min(W_float32.shape)
        k = min(self.max_k, max(self.min_k, int(math.floor(self.top_p * max_rank))))
        k = min(k, max_rank)

        print(f"    - Tensor shape: {list(W_float32.shape)}, Max rank: {max_rank}. Using k={k} components.")

        if self.full_matrix == True:
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

        if self.optimizer_choice == 'ppsf':
            final_tensor_scaled = self._optimize_ppsf(W_float32, scale, U_k, Vh_k)
            final_tensor_scaled.clamp_(-self.f8_max_val, self.f8_max_val)
        elif self.optimizer_choice == 'adamw':
            final_tensor_scaled = self._optimize_adamw(W_float32, scale, U_k, Vh_k)
            final_tensor_scaled.clamp_(-self.f8_max_val, self.f8_max_val)
        elif self.optimizer_choice == 'radam':
            final_tensor_scaled = self._optimize_radam(W_float32, scale, U_k, Vh_k)
            final_tensor_scaled.clamp_(-self.f8_max_val, self.f8_max_val)
        elif self.optimizer_choice == 'original':
            final_tensor_scaled = self._optimize_original(W_float32, scale, U_k, Vh_k)
            final_tensor_scaled.clamp_(-self.f8_max_val, self.f8_max_val)
        else:
            raise ValueError(f"Unknown optimizer: '{self.optimizer_choice}'")

        #    final_tensor_scaled = self._optimize_original(W_float32, scale, U_k, Vh_k)
        #    final_tensor_scaled.clamp_(-self.f8_max_val, self.f8_max_val)

        with torch.no_grad():
            W_f8 = final_tensor_scaled.to(TARGET_FP8_DTYPE)
            # Ensure compact_scale is valid before calling reciprocal; fall back to ones if missing.
            if compact_scale is None:
                print("    - WARNING: compact_scale is None, falling back to torch.ones for dequant_scale.")
                dequant_scale = torch.ones(1, device=self.device, dtype=SCALE_DTYPE)
            else:
                if current_scaling_mode == 'block':
                    dequant_scale = compact_scale.reciprocal()
                else:
                    dequant_scale = compact_scale.reciprocal().reshape(1)
                dequant_scale = dequant_scale.to(device=self.device, dtype=SCALE_DTYPE)
            dequantized_weight_tensor = (W_f8.to(self.device, dtype=COMPUTE_DTYPE) / scale)
        del W_float32, scale, U, Vh, U_k, Vh_k, final_tensor_scaled, compact_scale
        gc.collect()
        if self.device == 'cuda':
            torch.cuda.empty_cache()

        return W_f8, dequant_scale.to(device=self.device, dtype=SCALE_DTYPE), dequantized_weight_tensor

# --- Main script execution functions ---
def get_fp8_constants(fp8_dtype: torch.dtype) -> Tuple[float, float, float]:
    finfo = torch.finfo(fp8_dtype)
    return float(finfo.min), float(finfo.max), float(finfo.tiny)

FP8_MIN, FP8_MAX, FP8_MIN_POS = get_fp8_constants(TARGET_FP8_DTYPE)

def convert_to_fp8_scaled(
    input_file: str, output_file: str, t5xxl: bool, keep_distillation_large: bool,
    keep_distillation_small: bool, keep_nerf_large: bool, keep_nerf_small: bool,
    radiance: bool, wan: bool, qwen: bool, hunyuan: bool, zimage_l: bool, zimage_s: bool, calib_samples: int, seed: int,
    **converter_kwargs
):
    print(f"Processing: {input_file}\nOutput will be saved to: {output_file}")
    print("-" * 60)
    print(f"Target FP8 format: {TARGET_FP8_DTYPE}\nFP8 Range: [{FP8_MIN}, {FP8_MAX}]")
    print("-" * 60)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    seed_device = device
    seed_generator = torch.Generator(device=seed_device)
    seed_generator.manual_seed(seed)

    tensors: Dict[str, torch.Tensor] = {}
    try:
        with safe_open(input_file, framework="pt", device='cpu') as f:
            print(f"Loading {len(f.keys())} tensors from source file...")
            for key in tqdm(f.keys(), desc="Loading tensors"):
                tensors[key] = f.get_tensor(key)
    except Exception as e:
        print(f"FATAL: Error loading '{input_file}': {e}")
        return

    converter = LearnedRoundingConverter(**converter_kwargs)

    print("\nScanning model and generating simulated calibration data...")
    calibration_data_cache = {}
    for key, tensor in tensors.items():
        if key.endswith('.weight') and tensor.ndim == 2:
            in_features = tensor.shape[1]
            if in_features not in calibration_data_cache:
                print(f"  - Found new input dimension: {in_features}.")
                calibration_data_cache[in_features] = torch.randn(calib_samples, in_features, dtype=COMPUTE_DTYPE, generator=seed_generator, device=seed_device)
    print("Simulated calibration data generated.\n")

    new_tensors: Dict[str, torch.Tensor] = {}
    weight_keys = sorted([key for key in tensors.keys() if key.endswith('.weight') and tensors[key].ndim == 2])
    total_weights = len(weight_keys)
    skipped_count = 0
    processed_count = 0

    print(f"Found {total_weights} weight tensors to potentially process.")
    print("-" * 60)

    for i, key in enumerate(weight_keys):
        process_this_key = True
        create_scale_key = True
        skip_reason = ""

        if t5xxl and any(n in key for n in T5XXL_REMOVE_KEY_NAMES):
            print(f"({i+1}/{total_weights}) Removing T5XXL decoder tensor: {key}")
            skipped_count += 1
            continue
        if t5xxl and any(n in key for n in AVOID_KEY_NAMES):
            skip_reason = "T5XXL exclusion"
            create_scale_key = False
            process_this_key = False
        if radiance and any(n in key for n in RADIANCE_LAYER_KEYNAMES):
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
        if keep_distillation_large and any(n in key for n in DISTILL_LAYER_KEYNAMES_LARGE):
            skip_reason = "Distillation layer and Flux1 keep"
            create_scale_key = True
            process_this_key = False
        if keep_distillation_small and any(n in key for n in DISTILL_LAYER_KEYNAMES_SMALL):
            skip_reason = "Distillation layer only"
            create_scale_key = True
            process_this_key = False
        if keep_nerf_large and any(n in key for n in NERF_LAYER_KEYNAMES_LARGE):
            skip_reason = "Distillation layer, NeRF layer and txt_in"
            create_scale_key = True
            process_this_key = False
        if keep_nerf_small and any(n in key for n in NERF_LAYER_KEYNAMES_SMALL):
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

        if not process_this_key:
            if not create_scale_key:
                print(f"({i+1}/{total_weights}) Skipping tensor: {key} (Reason: {skip_reason})")
                new_tensors[key] = tensors[key].to(device='cpu')
                skipped_count += 1
                continue
            else:
                print(f"({i+1}/{total_weights}) Skipping tensor: {key} (Reason: {skip_reason})")
                new_tensors[key] = tensors[key]
                base_name = key[:key.rfind('.weight')]
                new_tensors[f"{base_name}.scale_weight"] = torch.tensor([1.0], dtype=SCALE_DTYPE).to(device='cpu')
                skipped_count += 1
                continue

        print(f"({i+1}/{total_weights}) Processing tensor: {key}")
        processed_count += 1
        original_tensor = tensors[key]

        if original_tensor.numel() == 0 or original_tensor.ndim != 2:
            print(f"  - Skipping empty or non-2D tensor: {key}")
            new_tensors[key] = original_tensor.to(TARGET_FP8_DTYPE)
            base_name = key[:key.rfind('.weight')]
            new_tensors[f"{base_name}.scale_weight"] = torch.tensor([1.0], dtype=SCALE_DTYPE).to(device='cpu')
            continue

        q_tensor, dequant_s, dequant_w = converter.convert(original_tensor)
        new_tensors[key] = q_tensor.to(device='cpu')
        base_name = key[:key.rfind('.weight')]
        bias_key = f"{base_name}.bias"
        new_tensors[f"{base_name}.scale_weight"] = dequant_s.to(device='cpu').detach().clone()

        if bias_key in tensors:
            print(f"  - Adjusting corresponding bias: {bias_key}")
            with torch.no_grad():
                original_bias = tensors[bias_key]
                in_features = original_tensor.shape[1]
                if in_features not in calibration_data_cache:
                    print(f"  - WARNING: No calibration data for bias correction.")
                    new_tensors[bias_key] = original_bias
                else:
                    X_calib_dev = calibration_data_cache[in_features].to(device=device)
                    W_orig_dev = original_tensor.to(device=device, dtype=COMPUTE_DTYPE)
                    W_dequant_dev = dequant_w.to(device=device, dtype=COMPUTE_DTYPE)
                    b_orig_dev = original_bias.to(device=device, dtype=COMPUTE_DTYPE)
                    weight_error = W_orig_dev - W_dequant_dev
                    output_error = X_calib_dev @ weight_error.T
                    bias_correction = output_error.mean(dim=0)
                    b_new = b_orig_dev - bias_correction
                    new_tensors[bias_key] = b_new.to(device='cpu', dtype=original_bias.dtype)
                    print(f"    - Original bias mean : {original_bias.mean().item():.6f}\n    - Corrected bias mean: {new_tensors[bias_key].mean().item():.6f}")
                    del W_orig_dev, W_dequant_dev, X_calib_dev, b_orig_dev, weight_error, output_error, bias_correction, b_new
                    if device == 'cuda': torch.cuda.empty_cache()

        if t5xxl:
            new_tensors[f"{base_name}.scale_input"] = dequant_s.to(device='cpu').detach().clone()

        print(f"    - Final Dequant Scale shape: {dequant_s.shape}\n    - Final Weight shape       : {q_tensor.shape}")
        print("-" * 60)

    for key, tensor in tensors.items():
        if (any(n in key for n in T5XXL_REMOVE_KEY_NAMES) and t5xxl):
            continue
        if key not in new_tensors:
            new_tensors[key] = tensor

    new_tensors["scaled_fp8"] = torch.empty((0), dtype=TARGET_FP8_DTYPE) if t5xxl else torch.empty((2), dtype=TARGET_FP8_DTYPE)

    print(f"Saving {len(new_tensors)} tensors to {output_file}")
    try:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        save_file(new_tensors, output_file)
        print("Conversion complete!")
    except Exception as e:
        print(f"FATAL: Error saving file '{output_file}': {e}")
        return

    print("-" * 60)
    print("Summary:")
    print(f"  - Original tensor count : {len(tensors)}\n  - Weights processed     : {processed_count}\n  - Weights skipped       : {skipped_count}\n  - Final tensor count    : {len(new_tensors)}")
    print("-" * 60)

def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter, description=f"Convert safetensors weights to Scaled {TARGET_FP8_DTYPE} format.")
    parser.add_argument("--input", type=str, required=True, help="Input safetensors file path.")
    parser.add_argument("--output", type=str, help="Output safetensors file path. Auto-generated if not provided.")
    parser.add_argument("--t5xxl", action='store_true', help="Apply exclusions for T5XXL Text Encoder models.")
    parser.add_argument("--keep_distillation_large", action='store_true', help="Exclude known distillation layers and other sensitive.")
    parser.add_argument("--keep_distillation_small", action='store_true', help="Exclude known distillation layers.")
    parser.add_argument("--keep_nerf_large", action='store_true', help="Exclude known NeRF layers, distillation layers and txt_in.")
    parser.add_argument("--keep_nerf_small", action='store_true', help="Exclude known NeRF layers and distillation layers.")
    parser.add_argument("--radiance", action='store_true', help="Exclude known Radiance Field layers.")
    parser.add_argument("--wan", action='store_true', help="Exclude known WAN layers.")
    parser.add_argument("--qwen", action='store_true', help="Exclude known Qwen Image layers.")
    parser.add_argument("--hunyuan", action='store_true', help="Exclude known Hunyuan Video 1.5 layers.")
    parser.add_argument("--zimage_l", action='store_true', help="Exclude known Z-Image layers.")
    parser.add_argument("--zimage_s", action='store_true', help="Exclude known Z-Image layers.")
    parser.add_argument("--full_matrix", type=bool, default=False, help="If should use torch.linalg.svd with full matices instead of the torch.svd_lowrank.")
    parser.add_argument("--scaling_mode", type=str, default="tensor", choices=["tensor", "block"], help="Quantization scaling mode.")
    parser.add_argument("--block_size", type=int, default=64, help="Block size for 'block' scaling mode.")
    parser.add_argument("--calib_samples", type=int, default=3072, help="Number of random samples for bias correction.")
    parser.add_argument("--manual_seed", type=int, default=42, help="Set a manual seed for reproducibility. Use -1 for random.")
    parser.add_argument("--optimizer", type=str, default="original", choices=["original", "adamw", "ppsf", "radam"], help="Optimization algorithm.")
    parser.add_argument("--num_iter", type=int, default=500, help="Total optimization iterations per tensor.")
    parser.add_argument("--lr", type=float, default=1e-2, help="[AdamW/RAdam/Original] Initial learning rate.")
    parser.add_argument("--top_p", type=float, default=0.01, help="Proportion of principal components (SVD) to use.")
    parser.add_argument("--min_k", type=int, default=1, help="Minimum number of principal components.")
    parser.add_argument("--max_k", type=int, default=16, help="Maximum number of principal components.")

    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Error: Input file not found: {args.input}")
        return

    try:
        _ = torch.zeros(1, dtype=TARGET_FP8_DTYPE, device='cuda' if torch.cuda.is_available() else 'cpu')
    except (RuntimeError, TypeError):
        print("Error: This hardware/PyTorch version does not support the target FP8 dtype.")
        return

    if not args.output:
        base = os.path.splitext(args.input)[0]
        fp8_str = TARGET_FP8_DTYPE.__str__().split('.')[-1]
        flags = "".join(["_t5" if args.t5xxl else "", "_nodist_l" if args.keep_distillation_large else "", "_nodist_s" if args.keep_distillation_small else "", "_nonerf_l" if args.keep_nerf_large else "", "_nonerf_s" if args.keep_nerf_small else "", "_norad" if args.radiance else ""])
        output_file = f"{base}_{fp8_str}_{args.scaling_mode}{flags}_k{args.min_k}-{args.max_k}_p{args.top_p}_lr{args.lr}.safetensors"
    else:
        output_file = args.output

    if os.path.abspath(args.input) == os.path.abspath(output_file):
        print("Error: Output file cannot be same as input.")
        return

    seed = int(torch.randint(0, 2**32 - 1, ()).item()) if args.manual_seed == -1 else args.manual_seed
    print(f"Using seed: {seed}")

    converter_kwargs = {k: v for k, v in vars(args).items() if k not in ['input', 'output', 't5xxl', 'keep_distillation_large', 'keep_distillation_small', 'keep_nerf_large', 'keep_nerf_small', 'radiance', 'wan', 'qwen', 'hunyuan', 'zimage_l', 'zimage_s', 'calib_samples', 'manual_seed']}

    convert_to_fp8_scaled(
        args.input, output_file, args.t5xxl, args.keep_distillation_large,
        args.keep_distillation_small, args.keep_nerf_large, args.keep_nerf_small,
        args.radiance, args.wan, args.qwen, args.hunyuan, args.zimage_l, args.zimage_s, args.calib_samples, seed,
        **converter_kwargs
    )

if __name__ == "__main__":
    main()
