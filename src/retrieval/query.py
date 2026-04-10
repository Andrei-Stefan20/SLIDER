"""High-level query interface over a FAISS index."""

from typing import Any

import faiss
import numpy as np
import torch

from src.models.sae import SparseAutoencoder
from src.retrieval.steering import steer_query


def search(
    index: faiss.Index,
    query_emb: np.ndarray,
    k: int = 10,
) -> tuple[np.ndarray, np.ndarray]:
    """Search the FAISS index for the k nearest neighbours.

    Args:
        index: A populated FAISS index (inner-product or L2).
        query_emb: Query embedding, shape ``(D,)`` or ``(1, D)``.
        k: Number of results to return.

    Returns:
        Tuple ``(distances, indices)`` each of shape ``(k,)``.
    """
    query = np.ascontiguousarray(query_emb, dtype=np.float32).reshape(1, -1)
    distances, indices = index.search(query, k)
    return distances[0], indices[0]


def search_with_sliders(
    index: faiss.Index,
    query_emb: np.ndarray,
    sae_model: SparseAutoencoder,
    slider_config: dict[int, float],
    k: int = 10,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply SLIDERS steering to the query, then search the FAISS index.

    For each feature id in ``slider_config`` the corresponding SAE decoder
    column is used as the steering direction.

    Args:
        index: A populated FAISS index.
        query_emb: Base query embedding, shape ``(D,)``.
        sae_model: Trained :class:`SparseAutoencoder`.  Its decoder weight
            columns define the feature directions.
        slider_config: Mapping of ``{feature_id: alpha_value}``.  Only
            features with non-zero alpha values need to be included.
        k: Number of results to return.

    Returns:
        Tuple ``(distances, indices)`` each of shape ``(k,)``.
    """
    if not slider_config:
        return search(index, query_emb, k=k)

    # Extract decoder weight matrix: shape (input_dim, hidden_dim)
    if sae_model.tied_weights:
        decoder_weight = sae_model.encoder.weight.detach().cpu().numpy().T
    else:
        decoder_weight = sae_model.decoder.weight.detach().cpu().numpy().T

    feature_ids = list(slider_config.keys())
    alphas = [slider_config[fid] for fid in feature_ids]
    # directions: (n_sliders, D)
    directions = decoder_weight[:, feature_ids].T

    steered = steer_query(query_emb, directions, alphas)
    return search(index, steered, k=k)
