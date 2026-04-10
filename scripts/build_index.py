"""CLI script for building a FAISS index from pre-extracted embeddings.

Usage:
    python scripts/build_index.py \\
        --embeddings data/processed/plantvillage_embeddings.npy \\
        --output data/processed/index.faiss
"""

import argparse
from pathlib import Path

import numpy as np
from sklearn.preprocessing import normalize

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

    print(f"Loading embeddings from: {args.embeddings}")
    embeddings = np.load(args.embeddings).astype(np.float32)
    print(f"  Shape: {embeddings.shape}")

    if not args.no_normalize:
        print("L2-normalising embeddings...")
        embeddings = normalize(embeddings, norm="l2")

    print("Building FAISS IndexFlatIP...")
    index = build_index(embeddings)
    save_index(index, args.output)
    print(f"Index saved -> {args.output}  ({index.ntotal} vectors, dim={index.d})")


if __name__ == "__main__":
    main()
