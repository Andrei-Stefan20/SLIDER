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
        sae: Trained :class:`~src.models.sae.SparseAutoencoder` (eval mode).
        embeddings: Float32 array of shape ``(N, input_dim)``.
        batch_size: Number of samples processed per forward pass.

    Returns:
        Float32 array of shape ``(N, hidden_dim)`` containing ReLU activations.
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
    """Fraction of SAE features that never activate across the dataset.

    A feature is considered dead if its activation is zero for every sample.

    Args:
        activations: Float32 array of shape ``(N, hidden_dim)``.

    Returns:
        Scalar in ``[0, 1]``.  0 means all features are alive.
    """
    ever_active = (activations > 0).any(axis=0)
    return float((~ever_active).mean())


def mean_activations_per_feature(activations: np.ndarray) -> np.ndarray:
    """Compute the mean activation value for each feature across the dataset.

    Args:
        activations: Float32 array of shape ``(N, hidden_dim)``.

    Returns:
        Float32 array of shape ``(hidden_dim,)``.
    """
    return activations.mean(axis=0)


def sparsity_per_sample(activations: np.ndarray) -> np.ndarray:
    """Compute the fraction of zero activations for each sample.

    Args:
        activations: Float32 array of shape ``(N, hidden_dim)``.

    Returns:
        Float32 array of shape ``(N,)`` where each value is the proportion
        of features that are inactive for that sample.
    """
    return (activations == 0).mean(axis=1).astype(np.float32)


def top_activating_features(
    activations: np.ndarray,
    image_idx: int,
    top_n: int = 10,
) -> list[int]:
    """Return the indices of the most activated features for a single image.

    Args:
        activations: Float32 array of shape ``(N, hidden_dim)``.
        image_idx: Row index of the query image.
        top_n: Number of top features to return.

    Returns:
        List of ``top_n`` feature indices sorted by descending activation,
        including only features with non-zero activation.
    """
    row = activations[image_idx]
    active_mask = row > 0
    if not active_mask.any():
        return []
    ranked = np.argsort(row)[::-1]
    return [int(i) for i in ranked[:top_n] if row[i] > 0]
