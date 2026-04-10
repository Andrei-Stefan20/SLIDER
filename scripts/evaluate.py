"""CLI script for evaluating retrieval quality (Recall@K + CLIP alignment).

Usage:
    python scripts/evaluate.py \\
        --embeddings data/processed/plantvillage_embeddings.npy \\
        --image-paths data/processed/plantvillage_image_paths.json \\
        --index data/processed/index.faiss \\
        --sae-model models/sae_best.pt \\
        --feature-names models/feature_names.json
"""

import argparse
import json
from pathlib import Path, PurePath

import numpy as np
import torch

from src.encoders.clip_encoder import CLIPEncoder
from src.evaluation.clip_alignment import batch_clip_alignment
from src.evaluation.recall_at_k import mean_recall_at_k
from src.models.sae import SparseAutoencoder
from src.naming.feature_namer import get_top_images, rank_features_by_variance
from src.retrieval.index import load_index
from src.retrieval.query import search


def build_same_class_ground_truth(image_paths: list[str]) -> list[list[int]]:
    """For each image, return the indices of all other images in the same class directory."""
    labels = [PurePath(p).parent.name for p in image_paths]
    label_to_indices: dict[str, list[int]] = {}
    for i, label in enumerate(labels):
        label_to_indices.setdefault(label, []).append(i)

    return [
        [j for j in label_to_indices[labels[i]] if j != i]
        for i in range(len(image_paths))
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate SLIDERS retrieval.")
    parser.add_argument("--embeddings", type=Path, required=True)
    parser.add_argument("--image-paths", type=Path, required=True)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--sae-model", type=Path, required=True)
    parser.add_argument("--feature-names", type=Path, default=None)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--n-align-features", type=int, default=20)
    args = parser.parse_args()

    embeddings = np.load(args.embeddings).astype(np.float32)
    image_paths = json.loads(args.image_paths.read_text())
    index = load_index(args.index)

    # Load SAE
    state = torch.load(args.sae_model, map_location="cpu")
    input_dim = embeddings.shape[1]
    hidden_dim = state["encoder.weight"].shape[0]
    sae = SparseAutoencoder(input_dim=input_dim, hidden_dim=hidden_dim)
    sae.load_state_dict(state)
    sae.eval()

    # ---------- Recall@K ----------
    print("Computing Recall@K (same-class ground truth)...")
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norm_embs = embeddings / np.where(norms > 0, norms, 1.0)
    ground_truth = build_same_class_ground_truth(image_paths)

    results = []
    for i in range(len(norm_embs)):
        _, retrieved = search(index, norm_embs[i], k=args.top_k + 1)
        retrieved_filtered = [r for r in retrieved.tolist() if r != i][:args.top_k]
        results.append({"retrieved": retrieved_filtered, "relevant": ground_truth[i]})

    recall = mean_recall_at_k(results, k_values=[1, 5, 10])
    print("\n=== Recall@K ===")
    for k, v in recall.items():
        print(f"  {k}: {v:.4f}")

    # ---------- CLIP Alignment ----------
    if args.feature_names is not None:
        print("\nComputing CLIP alignment scores...")
        feature_names_map: dict = json.loads(args.feature_names.read_text())

        with torch.no_grad():
            activations = sae.encode(torch.from_numpy(embeddings)).numpy()

        ranked = rank_features_by_variance(activations)[: args.n_align_features]
        clip_enc = CLIPEncoder()

        named_features = []
        for fid in ranked:
            name = feature_names_map.get(str(fid), f"Feature {fid}")
            fi = get_top_images(activations, image_paths, fid, k=args.top_k)
            named_features.append({"feature_id": fid, "name": name, "top_paths": fi.top_paths})

        alignment_scores = batch_clip_alignment(named_features, clip_enc)
        mean_align = np.mean(list(alignment_scores.values()))

        print("\n=== CLIP Alignment ===")
        for fid, score in sorted(alignment_scores.items(), key=lambda x: -x[1]):
            name = feature_names_map.get(str(fid), f"Feature {fid}")
            print(f"  [{fid:5d}] {name:40s}  {score:.4f}")
        print(f"\n  Mean alignment: {mean_align:.4f}")


if __name__ == "__main__":
    main()
