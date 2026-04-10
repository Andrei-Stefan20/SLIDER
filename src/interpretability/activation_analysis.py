"""Statistical analysis of SAE activations across a dataset."""

import numpy as np
import torch

from src.models.sae import SparseAutoencoder


def compute_activations(
    sae: SparseAutoencoder,
    embeddings: np.ndarray,
    batch_size: int = 512,
) -> np.ndarray:
    """Run the SAE encoder over all embeddings and return the activation matrix.

    Args:
        sae: Trained SAE in eval mode.
        embeddings: Float32 array of shape ``(N, input_dim)``.
        batch_size: Samples per forward pass.

    Returns:
        Float32 array of shape ``(N, hidden_dim)``.
    """
    sae.eval()
    results: list[np.ndarray] = []
    tensor = torch.from_numpy(embeddings.astype(np.float32))

    with torch.no_grad():
        for start in range(0, len(tensor), batch_size):
            batch = tensor[start : start + batch_size]
            h = sae.encode(batch)
            results.append(h.cpu().numpy())

    return np.concatenate(results, axis=0)


def dead_feature_ratio(activations: np.ndarray) -> float:
    """Fraction of features that never activate across the dataset."""
    active = (activations > 0).any(axis=0)
    return float((~active).mean())


def mean_activations_per_feature(activations: np.ndarray) -> np.ndarray:
    """Mean activation per feature across all samples. Shape: ``(hidden_dim,)``."""
    return activations.mean(axis=0)


def sparsity_per_sample(activations: np.ndarray) -> np.ndarray:
    """Fraction of zero activations per sample. Shape: ``(N,)``."""
    return (activations == 0).mean(axis=1).astype(np.float32)


def top_activating_features(
    activations: np.ndarray,
    image_idx: int,
    top_n: int = 10,
) -> list[int]:
    """Feature indices with the highest activations for a single image.

    Only non-zero activations are returned.
    """
    row = activations[image_idx]
    if not (row > 0).any():
        return []
    ranked = np.argsort(row)[::-1]
    return [int(i) for i in ranked[:top_n] if row[i] > 0]
