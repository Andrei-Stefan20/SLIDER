"""CLI script for building a FAISS index from pre-extracted embeddings.

Usage:
    python scripts/build_index.py \\
        --embeddings data/processed/plantvillage_embeddings.npy \\
        --output data/processed/index.faiss
"""

import argparse
from pathlib import Path

import numpy as np

from src.retrieval.index import build_index, save_index


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a FAISS index from a .npy embedding file."
    )
    parser.add_argument(
        "--embeddings", type=Path, required=True, help="Path to .npy embeddings."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/index.faiss"),
        help="Output path for the FAISS index.",
    )
    parser.add_argument(
        "--no-normalize",
        action="store_true",
        help="Skip L2 normalisation (use if embeddings are already normalised).",
    )
    args = parser.parse_args()

    embeddings = np.load(args.embeddings).astype(np.float32)
    print(f"Loaded {embeddings.shape} from {args.embeddings}")

    if not args.no_normalize:
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / np.where(norms > 0, norms, 1.0)

    index = build_index(embeddings)
    save_index(index, args.output)
    print(f"Index saved -> {args.output}  ({index.ntotal} vectors, dim={index.d})")


if __name__ == "__main__":
    main()
