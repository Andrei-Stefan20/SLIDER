"""Identifies the most and least activating images for each SAE feature."""

from pathlib import Path
from typing import NamedTuple

import numpy as np


class FeatureImages(NamedTuple):
    """Holds per-feature image paths and activation values."""

    feature_id: int
    top_paths: list[Path]
    top_activations: list[float]
    bottom_paths: list[Path]
    bottom_activations: list[float]


def get_top_images(
    activations: np.ndarray,
    image_paths: list[Path | str],
    feature_id: int,
    k: int = 10,
) -> FeatureImages:
    """Find the K images with the highest and lowest activations for a feature.

    Args:
        activations: Float32 array of shape ``(N, hidden_dim)`` containing
            per-image SAE activations.
        image_paths: Ordered list of image paths corresponding to rows of
            ``activations``.
        feature_id: Column index in ``activations`` to analyse.
        k: Number of top/bottom images to return per feature.

    Returns:
        :class:`FeatureImages` named tuple with sorted top and bottom images.
    """
    feature_acts = activations[:, feature_id]
    sorted_desc = np.argsort(feature_acts)[::-1]
    sorted_asc = np.argsort(feature_acts)

    top_idx = sorted_desc[:k].tolist()
    bottom_idx = sorted_asc[:k].tolist()

    paths = [Path(p) for p in image_paths]

    return FeatureImages(
        feature_id=feature_id,
        top_paths=[paths[i] for i in top_idx],
        top_activations=[float(feature_acts[i]) for i in top_idx],
        bottom_paths=[paths[i] for i in bottom_idx],
        bottom_activations=[float(feature_acts[i]) for i in bottom_idx],
    )


def rank_features_by_variance(activations: np.ndarray) -> list[int]:
    """Return feature indices sorted by activation variance (descending).

    High-variance features tend to capture more meaningful visual variation.

    Args:
        activations: Float32 array of shape ``(N, hidden_dim)``.

    Returns:
        List of feature indices, most variable first.
    """
    variances = activations.var(axis=0)
    return np.argsort(variances)[::-1].tolist()
