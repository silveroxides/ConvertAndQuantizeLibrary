"""
Command-line logic for quantization tools.
"""
import argparse
import torch
from safetensors import safe_open
from safetensors.torch import save_file
from .core.converter import LearnedRoundingConverter, quantize_model
from .utils import get_layer_filters, generate_output_filename
from convert_and_quantize.constants import (
    TARGET_FP8_DTYPE,
    COMPUTE_DTYPE,
    SCALE_DTYPE,
    T5XXL_REMOVE_KEY_NAMES,
)
def test_quantization(
    model_path: str,
    num_iter: int,
    exclude_layers: str,
    optimizer: str,
    block_size: int,
    full_matrix: bool,
    manual_seed: int,
    t5xxl: bool,
    chroma_large: bool,
    chroma_small: bool,
    radiance_large: bool,
    radiance_small: bool,
    radiance: bool,
    wan: bool,
    qwen: bool,
    hunyuan: bool,
    zimage_l: bool,
    zimage_s: bool,
):
    converter = LearnedRoundingConverter(
        optimizer=optimizer,
        num_iter=num_iter,
        scaling_mode="tensor", # test runner always uses tensor scaling
        block_size=block_size,
        full_matrix=full_matrix,
        seed=manual_seed,
    )

    quantized_tensors = quantize_model(
        model_path=model_path,
        converter=converter,
        exclude_layers=exclude_layers,
        manual_seed=manual_seed,
        t5xxl=t5xxl,
        chroma_large=chroma_large,
        chroma_small=chroma_small,
        radiance_large=radiance_large,
        radiance_small=radiance_small,
        radiance=radiance,
        wan=wan,
        qwen=qwen,
        hunyuan=hunyuan,
        zimage_l=zimage_l,
        zimage_s=zimage_s,
    )

    original_tensors = {}
    with safe_open(model_path, framework="pt", device="cpu") as f:
        for key in f.keys():
            original_tensors[key] = f.get_tensor(key)

    stats = []
    for key, q_tensor in quantized_tensors.items():
        if key in original_tensors and q_tensor.dtype == torch.float8_e4m3fn:
            o_tensor = original_tensors[key].to(converter.device)
            scale_key = key.replace(".weight", ".scale_weight")
            if scale_key not in quantized_tensors: # Handle cases where scale_weight might be missing
                continue
            scale = quantized_tensors[scale_key].to(converter.device)
            dequantized = (q_tensor.to(converter.device, dtype=torch.float32) / scale).to(o_tensor.dtype)
            error = (o_tensor - dequantized).abs().mean().item()
            stats.append((key, error, False))

    if stats:
        mean_error = sum(s[1] for s in stats) / len(stats)
        print(f"Mean Absolute Error: {mean_error:.6e}")

def quantize_and_save(
    model_path: str,
    num_iter: int,
    exclude_layers: str,
    output: str,
    scaling_mode: str,
    min_k: int,
    max_k: int,
    top_p: float,
    lr: float,
    optimizer: str,
    block_size: int,
    full_matrix: bool,
    manual_seed: int,
    calib_samples: int,
    t5xxl: bool,
    chroma_large: bool,
    chroma_small: bool,
    radiance_large: bool,
    radiance_small: bool,
    radiance: bool,
    wan: bool,
    qwen: bool,
    hunyuan: bool,
    zimage_l: bool,
    zimage_s: bool,
):
    converter = LearnedRoundingConverter(
        optimizer=optimizer,
        num_iter=num_iter,
        top_p=top_p,
        min_k=min_k,
        max_k=max_k,
        scaling_mode=scaling_mode,
        block_size=block_size,
        full_matrix=full_matrix,
        seed=manual_seed,
        lr=lr,
    )

    quantized_tensors = quantize_model(
        model_path=model_path,
        converter=converter,
        exclude_layers=exclude_layers,
        calib_samples=calib_samples,
        manual_seed=manual_seed,
        t5xxl=t5xxl,
        chroma_large=chroma_large,
        chroma_small=chroma_small,
        radiance_large=radiance_large,
        radiance_small=radiance_small,
        radiance=radiance,
        wan=wan,
        qwen=qwen,
        hunyuan=hunyuan,
        zimage_l=zimage_l,
        zimage_s=zimage_s,
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
    test_parser.add_argument("--t5xxl", action="store_true", help="Apply T5-XXL specific logic.")
    test_parser.add_argument("--chroma_large", action='store_true', help="Exclude known distillation layers and other sensitive.")
    test_parser.add_argument("--chroma_small", action='store_true', help="Exclude known distillation layers.")
    test_parser.add_argument("--radiance_large", action='store_true', help="Exclude known NeRF layers, distillation layers and txt_in.")
    test_parser.add_argument("--radiance_small", action='store_true', help="Exclude known NeRF layers and distillation layers.")
    test_parser.add_argument("--radiance", action='store_true', help="Exclude known Radiance Field layers.")
    test_parser.add_argument("--wan", action='store_true', help="Exclude known WAN layers.")
    test_parser.add_argument("--qwen", action='store_true', help="Exclude known Qwen Image layers.")
    test_parser.add_argument("--hunyuan", action='store_true', help="Exclude known Hunyuan Video 1.5 layers.")
    test_parser.add_argument("--zimage_l", action='store_true', help="Exclude known Z-Image layers.")
    test_parser.add_argument("--zimage_s", action='store_true', help="Exclude known Z-Image layers.")

    # Save command
    save_parser = subparsers.add_parser("save", help="Quantize a model and save the result to a new file.")
    save_parser.add_argument("model_path", help="Path to the .safetensors model file.")
    save_parser.add_argument("--output", type=str, default=None, help="Output file path. If not provided, it will be auto-generated.")
    save_parser.add_argument("--num-iter", type=int, default=50, help="Number of optimization iterations per tensor.")
    save_parser.add_argument("--exclude-layers", type=str, default="zimage", help=filter_help)
    save_parser.add_argument("--scaling-mode", type=str, default="tensor", choices=["tensor", "block"], help="Scaling mode for quantization.")
    save_parser.add_argument("--min-k", type=int, default=256, help="Minimum number of SVD components.")
    save_parser.add_argument("--max-k", type=int, default=768, help="Maximum number of SVD components.")
    save_parser.add_argument("--top-p", type=float, default=0.25, help="Proportion of SVD components to use.")
    save_parser.add_argument("--lr", type=float, default=8.098e-3, help="Learning rate for the optimizer.")
    save_parser.add_argument("--optimizer", type=str, default="original", choices=["original", "adamw", "radam", "ppsf"], help="Optimizer to use.")
    save_parser.add_argument("--block-size", type=int, default=64, help="Block size for 'block' scaling mode.")
    save_parser.add_argument("--full-matrix", action="store_true", help="Use full SVD matrix.")
    save_parser.add_argument("--manual-seed", type=int, default=-1, help="Manual seed for reproducibility.")
    save_parser.add_argument("--calib-samples", type=int, default=0, help="Number of calibration samples for bias correction (placeholder).")
    save_parser.add_argument("--t5xxl", action="store_true", help="Apply T5-XXL specific logic.")
    save_parser.add_argument("--chroma_large", action='store_true', help="Exclude known distillation layers and other sensitive.")
    save_parser.add_argument("--chroma_small", action='store_true', help="Exclude known distillation layers.")
    save_parser.add_argument("--radiance_large", action='store_true', help="Exclude known NeRF layers, distillation layers and txt_in.")
    save_parser.add_argument("--radiance_small", action='store_true', help="Exclude known NeRF layers and distillation layers.")
    save_parser.add_argument("--radiance", action='store_true', help="Exclude known Radiance Field layers.")
    save_parser.add_argument("--wan", action='store_true', help="Exclude known WAN layers.")
    save_parser.add_argument("--qwen", action='store_true', help="Exclude known Qwen Image layers.")
    save_parser.add_argument("--hunyuan", action='store_true', help="Exclude known Hunyuan Video 1.5 layers.")
    save_parser.add_argument("--zimage_l", action='store_true', help="Exclude known Z-Image layers.")
    save_parser.add_argument("--zimage_s", action='store_true', help="Exclude known Z-Image layers.")

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
            t5xxl=args.t5xxl,
            chroma_large=args.chroma_large,
            chroma_small=args.chroma_small,
            radiance_large=args.radiance_large,
            radiance_small=args.radiance_small,
            radiance=args.radiance,
            wan=args.wan,
            qwen=args.qwen,
            hunyuan=args.hunyuan,
            zimage_l=args.zimage_l,
            zimage_s=args.zimage_s,
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
            chroma_large=args.chroma_large,
            chroma_small=args.chroma_small,
            radiance_large=args.radiance_large,
            radiance_small=args.radiance_small,
            radiance=args.radiance,
            wan=args.wan,
            qwen=args.qwen,
            hunyuan=args.hunyuan,
            zimage_l=args.zimage_l,
            zimage_s=args.zimage_s,
        )

if __name__ == "__main__":
    main()

