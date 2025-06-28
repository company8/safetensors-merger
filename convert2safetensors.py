import torch
from safetensors.torch import save_file
import os


def convert_any_pytorch_model_to_safetensors(input_path, output_path=None):
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"{input_path} does not exist.")

    print(f"Loading: {input_path}")
    data = torch.load(input_path, map_location="cpu")

    # If it's a checkpoint with 'state_dict'
    if isinstance(data, dict):
        if "state_dict" in data:
            data = data["state_dict"]
        elif "model" in data:
            data = data["model"]

    # Flatten nested modules if needed
    if not isinstance(data, dict):
        raise TypeError("Loaded data is not a dictionary of tensors.")

    tensor_dict = {k: v.clone() for k, v in data.items() if isinstance(v, torch.Tensor)}

    if not tensor_dict:
        raise ValueError("No tensor data found in the input file.")

    if output_path is None:
        if input_path.endswith(".bin"):
            output_path = input_path.replace(".bin", ".safetensors")
        elif input_path.endswith(".pt"):
            output_path = input_path.replace(".pt", ".safetensors")
        elif input_path.endswith(".pth"):
            output_path = input_path.replace(".pth", ".safetensors")
        else:
            output_path = input_path + ".safetensors"

    print(f"Saving to: {output_path}")
    save_file(tensor_dict, output_path)
    print("Conversion complete.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Convert .pt/.pth/.bin PyTorch models to .safetensors format")
    parser.add_argument("file", help="Path to input .pt/.pth/.bin file")
    parser.add_argument("--output", help="Optional output path for .safetensors file")

    args = parser.parse_args()
    convert_any_pytorch_model_to_safetensors(args.file, args.output)
