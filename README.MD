# Safetensors Shard Merger
This script merges multiple sharded `.safetensors` files into a single safetensors file.  
It is useful if you have a model split into parts (such as `model-00001-of-00003.safetensors`, `model-00002-of-00003.safetensors`, etc.) and want to reconstruct the full model.
---

## Requirements
- Python 3.7 or later
- Safetensors
- NumPy 1.x (2.x may not work)
---

## Installation
1. Clone or download this repository.
2. Setup virtual environment (recommended):
    ```bash
   python -m venv venv
   venv/Scripts/activate.bat or ./venv/Scripts/Activate.ps1
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
---

## Usage
1. Place all your `.safetensors` shard files in the 'shards' folder.
2. Run the script:
    ```bash
    python merge.py or python ./merge.py
    ```
3. The merged file (e.g., merged_model.safetensors) will be created in 'output' folder.
---

## Strip metadata
1. Check for metadata:
    ```bash
    python stripmetadata.py your_model.safetensors
2. To remove metadata:
    ```bash
    python stripmetadata.py --strip your_model.safetensors
    ```
3. The your_model_stripped.safetensors model is saved with no metadata in the same folder.
---
## Example
Suppose you have these files:
```
model-00001-of-00003.safetensors
model-00002-of-00003.safetensors
model-00003-of-00003.safetensors
```
You’ll get:
```
merged_model.safetensors
```
---

## Credits
- https://github.com/NotTheStallion/reshard-safetensors
