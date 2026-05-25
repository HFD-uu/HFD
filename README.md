# HFD — Handwriting Fréchet Distance

Measures similarity between two sets of handwriting images using features from [AttentionHTR](https://github.com/dmitrijsk/AttentionHTR) and the Fréchet distance.

## Setup

### 1. Download model weights

The weights file (`AttentionHTR-Imgur5K.pth`) is not included in this repository due to its size. Download it from Google Drive and place it in the root of this directory (next to `HFD.py`):

**[Download weights from Google Drive](https://drive.google.com/drive/folders/1h6edewgRUTJPzI81Mn0eSsqItnk9RMeO)**

If the weights file is missing, `calculate()` will raise a `FileNotFoundError`.

### 2. Install dependencies

```bash
pip install torch torchvision pillow numpy
```

## Usage

```python
from HFD import calculate

score = calculate(generated_image_paths, reference_image_paths)
print(f"HFD: {score:.4f}")  # lower = more similar
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `image_paths_1` | — | Paths to generated images |
| `image_paths_2` | — | Paths to reference images |
| `model_path` | `AttentionHTR-Imgur5K.pth` | Path to model weights |
| `device` | `"auto"` | `"auto"`, `"cuda"`, or `"cpu"` |
| `batch_size` | `16` | Batch size for feature extraction |
