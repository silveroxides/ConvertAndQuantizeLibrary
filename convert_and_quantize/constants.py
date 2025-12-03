"""
Constants and configuration for the quantization module.
"""

import torch

# --- Target Data Types ---
TARGET_FP8_DTYPE = torch.float8_e4m3fn
COMPUTE_DTYPE = torch.float32
SCALE_DTYPE = torch.float32

# --- FP8 Constants ---
FP8_MIN = float(torch.finfo(TARGET_FP8_DTYPE).min)
FP8_MAX = float(torch.finfo(TARGET_FP8_DTYPE).max)
FP8_MIN_POS = float(torch.finfo(TARGET_FP8_DTYPE).tiny)

# --- Model-Specific Layer Exclusion Lists ---

# Generic layers to avoid across most models
AVOID_KEY_NAMES = [
    "norm", "bias", "embed_tokens", "shared", "patch_embedding",
    "audio_model.patch_embedding", "ref_conv", "control_adapter",
    "motion_encoder.enc.net_app", "face_encoder.conv", "pose_patch_embedding",
    "motion_encoder.enc.fc", "img_emb.proj", "q_norm", "motion_encoder.dec",
    "head.modulation", "casual_audio_encoder", "cond_encoder", "frame_packer",
    "norm_k", "norm_q"
]

# T5-XXL specific exclusions
T5XXL_REMOVE_KEY_NAMES = ["decoder", "lm_head"]

# Qwen specific exclusions
QWEN_AVOID_KEY_NAMES = ["norm_added_k", "norm_added_q", "norm_k", "norm_q", "txt_norm"]

# Hunyuan Video specific exclusions
HUNYUAN_AVOID_KEY_NAMES = [
    "layernorm", "img_attn_k_norm", "img_attn_q_norm", "txt_attn_k_norm",
    "txt_attn_q_norm", "norm1", "norm2", "vision_in.proj.0", "vision_in.proj.4",
    "img_in.proj", "cond_type_embedding"
]

# Z-Image specific exclusions
ZIMAGE_AVOID_KEY_NAMES = [
    "cap_embedder.0", "cap_pad_token", "attention_norm1", "attention_norm2",
    "ffn_norm1", "ffn_norm2", "k_norm", "q_norm", "x_pad_token"
]

# Radiance field exclusions (completely skip, no scale_weight)
RADIANCE_AVOID_KEY_NAMES = ["img_in_patch", "nerf_final_layer_conv"]

# --- Model-Specific Layer Preservation Lists ---
# These layers are kept in high precision (scale_weight = 1.0, not quantized)

# Distillation layers for Chroma model
CHROMA_LAYER_KEYNAMES_LARGE = ["distilled_guidance_layer", "final_layer", "img_in", "txt_in"]
CHROMA_LAYER_KEYNAMES_SMALL = ["distilled_guidance_layer"]

# Radiance layers (variants for different configurations)
RADIANCE_LAYER_KEYNAMES_LARGE = ["distilled_guidance_layer", "nerf_blocks", "nerf_image_embedder", "txt_in"]
RADIANCE_LAYER_KEYNAMES_SMALL = ["distilled_guidance_layer", "nerf_blocks", "nerf_image_embedder"]

# WAN (Waymo) layers
WAN_LAYER_KEYNAMES = [
    "text_embedding", "time_embedding", "audio_model.text_embedding",
    "audio_model.time_embedding", "time_projection", "video_model.time_projection",
    "head.head", "face_encoder.out_proj", "face_adapter"
]

# Qwen specific layers
QWEN_LAYER_KEYNAMES = ["time_text_embed", "img_in", "norm_out", "proj_out", "txt_in"]

# Z-Image specific layers
ZIMAGE_LAYER_KEYNAMES = ["x_embedder", "final_layer", "cap_embedder.1", "adaLN_modulation", "t_embedder"]
