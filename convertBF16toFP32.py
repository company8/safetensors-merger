from safetensors.torch import load_file, save_file
import torch
import os
import sys


def convert_bf16_to_fp32(input_path, output_path=None):
    if not os.path.exists(input_path):
        print(f"Error: File not found -> {input_path}")
        return

    print(f"Loading: {input_path}")
    try:
        tensors = load_file(input_path)
    except Exception as e:
        print(f"Failed to load file: {e}")
        return

    print("Converting tensors from BF16 to FP32 (where applicable)...")
    fp32_tensors = {k: v.to(torch.float32) if v.dtype == torch.bfloat16 else v for k, v in tensors.items()}

    if output_path is None:
        output_path = input_path.replace(".safetensors", "_fp32.safetensors")

    try:
        save_file(fp32_tensors, output_path)
        print(f"Saved FP32 model to: {output_path}")
    except Exception as e:
        print(f"Failed to save file: {e}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python convert_bf16_to_fp32.py <input_file.safetensors> [output_file.safetensors]")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None

    convert_bf16_to_fp32(input_file, output_file)
