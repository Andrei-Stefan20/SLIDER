"""FAISS index construction, persistence, and loading."""

from pathlib import Path

import faiss
import numpy as np


def build_index(embeddings: np.ndarray) -> faiss.Index:
    """Build a FAISS inner-product index (equivalent to cosine similarity
    when embeddings are L2-normalised).

    Args:
        embeddings: Float32 array of shape ``(N, D)``.  Should be
            L2-normalised before calling this function if cosine similarity
            is desired.

    Returns:
        A populated :class:`faiss.IndexFlatIP` index.
    """
    embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    return index


def save_index(index: faiss.Index, path: Path | str) -> None:
    """Persist a FAISS index to disk.

    Args:
        index: Populated FAISS index.
        path: Destination file path (conventional extension: ``.faiss``).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(path))


def load_index(path: Path | str) -> faiss.Index:
    """Load a FAISS index from disk.

    Args:
        path: Path to a previously saved ``.faiss`` file.

    Returns:
        The loaded FAISS index, ready for search.
    """
    return faiss.read_index(str(path))
