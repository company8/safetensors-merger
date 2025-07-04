from safetensors.torch import load_file, save_file
import torch
import os

def safetensors_fp32_to_bf16(input_file, output_file=None):
    if not os.path.isfile(input_file):
        print(f"Error: File not found -> {input_file}")
        return

    print(f"Loading: {input_file}")
    tensors = load_file(input_file)

    bf16_tensors = {}
    for k, v in tensors.items():
        if isinstance(v, torch.Tensor):
            print(f"Converting: {k} ({v.dtype})")
            if v.dtype == torch.float32:
                bf16_tensors[k] = v.to(torch.bfloat16)
            else:
                bf16_tensors[k] = v
        else:
            print(f"Skipping non-tensor: {k}")

    output_file = output_file or input_file.replace(".safetensors", "-BF16.safetensors")
    print(f"Saving to: {output_file}")
    save_file(bf16_tensors, output_file)
    print("Conversion complete.")


# Example usage
if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python convert_fp32_to_bf16.py <input_file.safetensors> [output_file.safetensors]")
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else None
    safetensors_fp32_to_bf16(input_path, output_path)
