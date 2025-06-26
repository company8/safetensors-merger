from safetensors.torch import load_file, save_file
from safetensors import safe_open
import sys
import os

# === View metadata ===
def print_metadata(safetensors_path):
    if not os.path.exists(safetensors_path):
        print(f"File not found: {safetensors_path}")
        return

    try:
        with safe_open(safetensors_path, framework="pt") as f:
            metadata = f.metadata()
        print("Metadata:")
        if metadata:
            for key, value in metadata.items():
                print(f"{key}: {value}")
        else:
            print("No metadata found.")
    except Exception as e:
        print(f"Error reading metadata: {e}")

# === Strip metadata ===
def strip_metadata(safetensors_path, output_path=None):
    if not os.path.exists(safetensors_path):
        print(f"File not found: {safetensors_path}")
        return

    try:
        tensors = load_file(safetensors_path)
        output_path = output_path or safetensors_path.replace(".safetensors", "_stripped.safetensors")
        save_file(tensors, output_path, metadata={})
        print(f"Stripped metadata and saved to {output_path}")
    except Exception as e:
        print(f"Error stripping metadata: {e}")

# === Example Usage ===
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="View or strip metadata from a .safetensors file")
    parser.add_argument("file", help="Path to the .safetensors file")
    parser.add_argument("--strip", action="store_true", help="Strip metadata and save new file")

    args = parser.parse_args()

    if args.strip:
        strip_metadata(args.file)
    else:
        print_metadata(args.file)
