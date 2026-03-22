# Aksharam

## Configuration

Tool paths (Tesseract, Poppler) and input/output filenames are stored in a local
configuration module that is **not** checked in to source control.

1. Copy `config_local.py` (provided as a template) to the workspace root.
2. Edit the constants to point at your installed tools and any custom filenames.
3. `config_local.py` is listed in `.gitignore` so it won't be committed.

The scripts will automatically fall back to sensible defaults if the config is
missing.

## CUDA GPU Acceleration

This project supports NVIDIA CUDA acceleration for significant performance improvements
in machine learning tasks (LaBSE embeddings and mBART translation).

### Quick Setup

1. **Check CUDA status**: `python cuda_setup.py`
2. **Install dependencies**: `pip install -r requirements.txt`
3. **Run with GPU**: The code automatically detects and uses CUDA when available

### Full CUDA Installation

For maximum performance with NVIDIA GPUs:

1. Install NVIDIA drivers from: https://www.nvidia.com/Download/index.aspx
2. Install CUDA Toolkit 11.8+ from: https://developer.nvidia.com/cuda-downloads
3. Install cuDNN from: https://developer.nvidia.com/cudnn
4. Install CUDA PyTorch: `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118`

### Performance Benefits

- **LaBSE Embeddings**: 5-10x faster on GPU vs CPU
- **mBART Translation**: 3-8x faster on GPU vs CPU
- **Memory Efficiency**: Better batch processing capabilities

The project automatically falls back to CPU if CUDA is not available.
