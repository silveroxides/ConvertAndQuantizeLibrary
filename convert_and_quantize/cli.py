"""
Command-line logic for quantization tools.
"""
import torch
from safetensors.torch import safe_open, save_file
from convert_and_quantize import LearnedRoundingConverter
from convert_and_quantize.utils import get_layer_filters, generate_output_filename

# --- Target Data Types ---
TARGET_FP8_DTYPE = torch.float8_e4m3fn
COMPUTE_DTYPE = torch.float32
SCALE_DTYPE = torch.float32

def test_quantization(model_path: str, num_iter: int, exclude_layers: str):
    """
    Test quantization on a model and report error metrics.
    """
    all_avoid_keys, layer_keys = get_layer_filters(exclude_layers)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    print(f"Using exclusion filter: {exclude_layers}")
    
    converter = LearnedRoundingConverter(
        optimizer="original",
        num_iter=num_iter,
        top_p=0.1,
        min_k=256,
        max_k=768,
        scaling_mode="tensor",
        lr=8.0980000000000011e-3,
    )
    
    quantized_count = 0
    skipped_count = 0
    total_tensors = 0
    stats = []
    
    with safe_open(model_path, framework="pt", device="cpu") as f:
        keys = list(f.keys())
        for key in keys:
            total_tensors += 1
            tensor = f.get_tensor(key)
            
            if not key.endswith(".weight") or tensor.ndim != 2:
                skipped_count += 1
                continue
            
            if any(n in key for n in all_avoid_keys):
                skipped_count += 1
                continue
            
            if any(n in key for n in layer_keys):
                stats.append((key, 0.0, True))
                quantized_count += 1
                skipped_count += 1
                continue
            
            t = tensor.to(device)
            q, scale, deq = converter.convert(t)
            error = (t.to(dtype=deq.dtype, device=deq.device) - deq).abs().mean().item()
            stats.append((key, error, False))
            quantized_count += 1
            
    print(f"Total tensors seen: {total_tensors}")
    print(f"Quantized tensors: {quantized_count}")
    print(f"Skipped tensors: {skipped_count}")
    
    if stats:
        mean_error = sum(s[1] for s in stats) / len(stats)
        high_precision_count = sum(1 for s in stats if s[2])
        print(f"Mean Absolute Error: {mean_error:.6e}")
        print(f"High precision tensors: {high_precision_count}")

def quantize_and_save(model_path: str, num_iter: int, exclude_layers: str, output: str, scaling_mode: str, min_k: int, max_k: int, top_p: float, lr: float):
    """
    Quantize a model and save the result.
    """
    all_avoid_keys, layer_keys = get_layer_filters(exclude_layers)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    converter = LearnedRoundingConverter(
        optimizer="original",
        num_iter=num_iter,
        top_p=top_p,
        min_k=min_k,
        max_k=max_k,
        scaling_mode=scaling_mode,
        lr=lr,
    )
    
    new_tensors = {}
    with safe_open(model_path, framework="pt", device="cpu") as f:
        for key in f.keys():
            tensor = f.get_tensor(key)
            
            if not key.endswith(".weight") or tensor.ndim != 2:
                new_tensors[key] = tensor
                continue
            
            if any(n in key for n in all_avoid_keys):
                new_tensors[key] = tensor
                continue
            
            if any(n in key for n in layer_keys):
                new_tensors[key] = tensor
                base_name = key[:key.rfind('.weight')]
                new_tensors[f"{base_name}.scale_weight"] = torch.tensor([1.0], dtype=SCALE_DTYPE)
                continue
            
            q_tensor, dequant_s, dequant_w = converter.convert(tensor)
            new_tensors[key] = q_tensor.to(device='cpu')
            base_name = key[:key.rfind('.weight')]
            new_tensors[f"{base_name}.scale_weight"] = dequant_s.to(device='cpu').detach().clone()
            
    if not output:
        output = generate_output_filename(
            model_path,
            TARGET_FP8_DTYPE,
            scaling_mode,
            exclude_layers,
            min_k,
            max_k,
            top_p,
            lr
        )
        
    save_file(new_tensors, output)
    print(f"Saved quantized model to: {output}")

