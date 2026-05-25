import os
import sys

# attentionHTR.py imports tools.pairing from the mFID research repo
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "mFID", "src"))

import numpy as np
import torch
from attentionHTR import _extract_from_paths, load_model

DEFAULT_MODEL_PATH = os.path.join(os.path.dirname(__file__), "AttentionHTR-Imgur5K.pth")


def calculate(
    image_paths_1: list[str],
    image_paths_2: list[str],
    model_path: str = DEFAULT_MODEL_PATH,
    device: str = "auto",
    batch_size: int = 16,
) -> float:
    """
    Calculate HFD between two sets of handwriting images.

    Args:
        image_paths_1: File paths to the first set of images (e.g. generated).
        image_paths_2: File paths to the second set of images (e.g. reference).
        model_path:    Path to AttentionHTR-Imgur5K.pth (defaults to bundled weights).
        device:        "auto", "cuda", or "cpu".
        batch_size:    Batch size for feature extraction.

    Returns:
        float: HFD score — lower means more similar.

    Example:
        from HFD import calculate
        score = calculate(generated_paths, reference_paths)
    """
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = load_model(model_path, device=device)
    model.eval()

    # (N, T, D) — sequential contextual features
    features1 = _extract_from_paths(image_paths_1, model, device, batch_size)
    features2 = _extract_from_paths(image_paths_2, model, device, batch_size)

    # mean pool over time: (N, T, D) -> (N, D)
    features1 = features1.mean(axis=1)
    features2 = features2.mean(axis=1)

    return _frechet_distance(features1, features2, device)


def _frechet_distance(
    pred: np.ndarray,
    target: np.ndarray,
    device: str,
    eps: float = 1e-6,
) -> float:
    pred_t = torch.as_tensor(pred, device=device, dtype=torch.float64)
    target_t = torch.as_tensor(target, device=device, dtype=torch.float64)
    n = pred_t.shape[0]

    pred_mean = pred_t.mean(0)
    target_mean = target_t.mean(0)
    mean_term = ((pred_mean - target_mean) ** 2).sum()

    eye = torch.eye(pred_t.shape[1], device=device, dtype=torch.float64) * eps
    pred_cov = (pred_t - pred_mean).T @ (pred_t - pred_mean) / max(n - 1, 1) + eye
    target_cov = (target_t - target_mean).T @ (target_t - target_mean) / max(n - 1, 1) + eye

    eigvals = torch.linalg.eigvals(pred_cov @ target_cov).real.clamp(min=0)
    trace_sqrt = torch.sqrt(eigvals).sum()
    return float(mean_term + pred_cov.trace() + target_cov.trace() - 2 * trace_sqrt)
