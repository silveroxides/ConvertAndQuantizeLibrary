"""
Command-line logic for quantization tools.
"""
import torch
from safetensors.torch import safe_open, save_file
from convert_and_quantize import LearnedRoundingConverter
from convert_and_quantize.utils import get_layer_filters, generate_output_filename
from convert_and_quantize.constants import T5XXL_REMOVE_KEY_NAMES
import argparse

# --- Target Data Types ---
TARGET_FP8_DTYPE = torch.float8_e4m3fn
COMPUTE_DTYPE = torch.float32
SCALE_DTYPE = torch.float32

def test_quantization(model_path: str, num_iter: int, exclude_layers: str, optimizer: str, block_size: int, full_matrix: bool, manual_seed: int):
    from convert_and_quantize.core.converter import quantize_model
    from safetensors.torch import safe_open

    converter = LearnedRoundingConverter(
        optimizer=optimizer,
        num_iter=num_iter,
        scaling_mode="tensor", # test runner always uses tensor scaling
        block_size=block_size,
        full_matrix=full_matrix,
    )

    with safe_open(model_path, framework="pt", device="cpu") as f:
        original_tensors = {k: f.get_tensor(k) for k in f.keys()}
    quantized_tensors = quantize_model(
        model_path=model_path,
        converter=converter,
        exclude_layers=exclude_layers,
    )

    stats = []
    for key, q_tensor in quantized_tensors.items():
        if key in original_tensors and q_tensor.dtype == torch.float8_e4m3fn:
            o_tensor = original_tensors[key].to(converter.device)
            scale_key = key.replace(".weight", ".scale_weight")
            scale = quantized_tensors[scale_key].to(converter.device)
            dequantized = (q_tensor.to(converter.device, dtype=torch.float32) / scale).to(o_tensor.dtype)
            error = (o_tensor - dequantized).abs().mean().item()
            stats.append((key, error, False))

    if stats:
        mean_error = sum(s[1] for s in stats) / len(stats)
        print(f"Mean Absolute Error: {mean_error:.6e}")

def quantize_and_save(model_path: str, num_iter: int, exclude_layers: str, output: str, scaling_mode: str, min_k: int, max_k: int, top_p: float, lr: float, optimizer: str, block_size: int, full_matrix: bool, manual_seed: int, calib_samples: int, t5xxl: bool):
    from convert_and_quantize.core.converter import quantize_model
    from safetensors.torch import save_file

    converter = LearnedRoundingConverter(
        optimizer=optimizer,
        num_iter=num_iter,
        top_p=top_p,
        min_k=min_k,
        max_k=max_k,
        scaling_mode=scaling_mode,
        block_size=block_size,
        full_matrix=full_matrix,
        lr=lr,
    )

    quantized_tensors = quantize_model(
        model_path=model_path,
        converter=converter,
        exclude_layers=exclude_layers,
        calib_samples=calib_samples,
        manual_seed=manual_seed,
        t5xxl=t5xxl,
    )
    
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
        
    save_file(quantized_tensors, output)
    print(f"Saved quantized model to: {output}")

def main():
    parser = argparse.ArgumentParser(
        description="A suite of tools for quantizing neural networks with learned adaptive rounding.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Common arguments
    filter_choices = ['zimage', 'qwen', 'hunyuan', 'chroma_l', 'chroma_s', 'nerf_l', 'nerf_s', 'radiance', 'wan']
    filter_help = (
        "Layer exclusion filter preset.\n"
        "Pre-defined choices: \n"
        f"  {', '.join(filter_choices)}\n"
        "Can also be a custom comma-separated list of layer names to exclude."
    )

    # Test command
    test_parser = subparsers.add_parser("test", help="Test quantization and report error metrics without saving.")
    test_parser.add_argument("model_path", help="Path to the .safetensors model file.")
    test_parser.add_argument("--num-iter", type=int, default=50, help="Number of optimization iterations per tensor.")
    test_parser.add_argument("--exclude-layers", type=str, default="zimage", help=filter_help)
    test_parser.add_argument("--optimizer", type=str, default="original", choices=["original", "adamw", "radam"], help="Optimizer to use.")
    test_parser.add_argument("--block-size", type=int, default=64, help="Block size for 'block' scaling mode.")
    test_parser.add_argument("--full-matrix", action="store_true", help="Use full SVD matrix.")
    test_parser.add_argument("--manual-seed", type=int, default=-1, help="Manual seed for reproducibility.")

    # Save command
    save_parser = subparsers.add_parser("save", help="Quantize a model and save the result to a new file.")
    save_parser.add_argument("model_path", help="Path to the .safetensors model file.")
    save_parser.add_argument("--output", type=str, default=None, help="Output file path. If not provided, it will be auto-generated.")
    save_parser.add_argument("--num-iter", type=int, default=50, help="Number of optimization iterations per tensor.")
    save_parser.add_argument("--exclude-layers", type=str, default="zimage", help=filter_help)
    save_parser.add_argument("--scaling-mode", type=str, default="tensor", choices=["tensor", "block"], help="Scaling mode for quantization.")
    save_parser.add_argument("--min-k", type=int, default=256, help="Minimum number of SVD components.")
    save_parser.add_argument("--max-k", type=int, default=768, help="Maximum number of SVD components.")
    save_parser.add_argument("--top-p", type=float, default=0.1, help="Proportion of SVD components to use.")
    save_parser.add_argument("--lr", type=float, default=8.098e-3, help="Learning rate for the optimizer.")
    save_parser.add_argument("--optimizer", type=str, default="original", choices=["original", "adamw", "radam", "ppsf"], help="Optimizer to use.")
    save_parser.add_argument("--block-size", type=int, default=64, help="Block size for 'block' scaling mode.")
    save_parser.add_argument("--full-matrix", action="store_true", help="Use full SVD matrix.")
    save_parser.add_argument("--manual-seed", type=int, default=-1, help="Manual seed for reproducibility.")
    save_parser.add_argument("--calib-samples", type=int, default=0, help="Number of calibration samples for bias correction (placeholder).")
    save_parser.add_argument("--t5xxl", action="store_true", help="Apply T5-XXL specific logic.")

    args = parser.parse_args()

    if args.command == "test":
        test_quantization(
            model_path=args.model_path,
            num_iter=args.num_iter,
            exclude_layers=args.exclude_layers,
            optimizer=args.optimizer,
            block_size=args.block_size,
            full_matrix=args.full_matrix,
            manual_seed=args.manual_seed,
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
            optimizer=args.optimizer,
            block_size=args.block_size,
            full_matrix=args.full_matrix,
            manual_seed=args.manual_seed,
            calib_samples=args.calib_samples,
            t5xxl=args.t5xxl,
        )

if __name__ == "__main__":
    main()

