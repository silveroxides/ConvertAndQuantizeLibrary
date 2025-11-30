"""
Command-line logic for quantization tools.
"""
import torch
from safetensors.torch import safe_open, save_file
from convert_and_quantize import LearnedRoundingConverter
from convert_and_quantize.utils import get_layer_filters, generate_output_filename
import argparse

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

def main():
    parser = argparse.ArgumentParser(description="Quantization tools")
    subparsers = parser.add_subparsers(dest="command")

    # Test command
    test_parser = subparsers.add_parser("test", help="Test quantization on a model")
    test_parser.add_argument("model_path", help="Path to the .safetensors model file")
    test_parser.add_argument("--num-iter", type=int, default=50, help="Number of iterations")
    test_parser.add_argument("--exclude-layers", type=str, default="zimage", help="Layer exclusion filter")

    # Save command
    save_parser = subparsers.add_parser("save", help="Quantize and save a model")
    save_parser.add_argument("model_path", help="Path to the .safetensors model file")
    save_parser.add_argument("--num-iter", type=int, default=50, help="Number of iterations")
    save_parser.add_argument("--exclude-layers", type=str, default="zimage", help="Layer exclusion filter")
    save_parser.add_argument("--output", type=str, default=None, help="Output file path")
    save_parser.add_argument("--scaling-mode", type=str, default="tensor", help="Scaling mode")
    save_parser.add_argument("--min-k", type=int, default=256, help="Minimum k")
    save_parser.add_argument("--max-k", type=int, default=768, help="Maximum k")
    save_parser.add_argument("--top-p", type=float, default=0.1, help="Top p")
    save_parser.add_argument("--lr", type=float, default=8.098e-3, help="Learning rate")

    args = parser.parse_args()

    if args.command == "test":
        test_quantization(
            model_path=args.model_path,
            num_iter=args.num_iter,
            exclude_layers=args.exclude_layers,
        )
    elif args.command == "save":
        quantize_and_save(
            model_path=args.model_path,
            num_iter=args.num_iter,
            exclude_layers=args.exclude_layers,
            output=args.output,
            scaling_mode=args.scaling_mode,
            min_k=args.min_k,
            max_k=args.max_k,
            top_p=args.top_p,
            lr=args.lr,
        )

if __name__ == "__main__":
    main()

