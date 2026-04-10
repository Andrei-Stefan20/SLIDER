"""Serialisation helpers for embeddings, image paths, and feature names."""

import json
from pathlib import Path

import numpy as np


def save_embeddings(embeddings: np.ndarray, path: Path | str) -> None:
    """Save a float32 embedding matrix to a ``.npy`` file.

    Args:
        embeddings: Array of shape ``(N, D)``.
        path: Destination file path.  Parent directories are created
            automatically.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, embeddings.astype(np.float32))


def load_embeddings(path: Path | str) -> np.ndarray:
    """Load a ``.npy`` embedding matrix and cast to float32.

    Args:
        path: Path to a ``.npy`` file of shape ``(N, D)``.

    Returns:
        Float32 NumPy array.
    """
    return np.load(path).astype(np.float32)


def save_image_paths(paths: list[str], path: Path | str) -> None:
    """Persist an ordered list of image path strings as JSON.

    Args:
        paths: List of absolute or relative path strings.
        path: Destination ``.json`` file path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(paths, indent=2))


def load_image_paths(path: Path | str) -> list[str]:
    """Load an ordered list of image path strings from a JSON file.

    Args:
        path: Path to a JSON file produced by :func:`save_image_paths`.

    Returns:
        List of path strings in the original order.
    """
    return json.loads(Path(path).read_text())


def save_feature_names(names_dict: dict[str | int, str], path: Path | str) -> None:
    """Persist a feature-id → name mapping as a JSON file.

    Args:
        names_dict: Mapping of feature id (int or str) to name string.
        path: Destination ``.json`` file path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Normalise keys to strings for JSON compatibility
    serialisable = {str(k): v for k, v in names_dict.items()}
    path.write_text(json.dumps(serialisable, indent=2))


def load_feature_names(path: Path | str) -> dict[str, str]:
    """Load a feature-id → name mapping from a JSON file.

    Args:
        path: Path to a JSON file produced by :func:`save_feature_names`.

    Returns:
        Dict with string keys (feature ids) and string values (names).
    """
    return json.loads(Path(path).read_text())
