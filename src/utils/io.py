"""Serialisation helpers for embeddings, image paths, and feature names."""

import json
from pathlib import Path

import numpy as np


def save_embeddings(embeddings: np.ndarray, path: Path | str) -> None:
    """Save a float32 embedding matrix to a .npy file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, embeddings.astype(np.float32))


def load_embeddings(path: Path | str) -> np.ndarray:
    """Load a .npy embedding matrix, cast to float32."""
    return np.load(path).astype(np.float32)


def save_image_paths(paths: list[str], path: Path | str) -> None:
    """Persist an ordered list of image path strings as JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(paths, indent=2))


def load_image_paths(path: Path | str) -> list[str]:
    """Load an ordered list of image path strings from a JSON file."""
    return json.loads(Path(path).read_text())


def save_feature_names(names_dict: dict[str | int, str], path: Path | str) -> None:
    """Persist a feature-id → name mapping as JSON. Keys are normalised to strings."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    out = {str(k): v for k, v in names_dict.items()}
    path.write_text(json.dumps(out, indent=2))


def load_feature_names(path: Path | str) -> dict[str, str]:
    """Load a feature-id → name mapping from a JSON file."""
    return json.loads(Path(path).read_text())
