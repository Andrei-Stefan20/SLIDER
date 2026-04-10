"""CLI script for naming SAE features using CLIP descriptions + an LLM.

Usage:
    python scripts/name_features.py \\
        --embeddings data/processed/plantvillage_embeddings.npy \\
        --image-paths data/processed/plantvillage_image_paths.json \\
        --sae-model models/sae_best.pt \\
        --output models/feature_names.json \\
        --n-features 20
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from src.models.sae import SparseAutoencoder
from src.naming.clip_describer import CLIPDescriber
from src.naming.feature_namer import get_top_images, rank_features_by_variance
from src.naming.llm_namer import LLMFeatureNamer


def main() -> None:
    parser = argparse.ArgumentParser(description="Name SAE features with LLM assistance.")
    parser.add_argument("--embeddings", type=Path, required=True)
    parser.add_argument("--image-paths", type=Path, required=True)
    parser.add_argument("--sae-model", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("models/feature_names.json"))
    parser.add_argument("--n-features", type=int, default=20, help="Number of features to name.")
    parser.add_argument("--top-k", type=int, default=10, help="Top/bottom K images per feature.")
    parser.add_argument("--llm-model", type=str, default="gpt-4o")
    args = parser.parse_args()

    embeddings = np.load(args.embeddings).astype(np.float32)
    image_paths = json.loads(args.image_paths.read_text())

    # Load SAE
    state = torch.load(args.sae_model, map_location="cpu")
    input_dim = embeddings.shape[1]
    hidden_dim = state["encoder.weight"].shape[0]
    sae = SparseAutoencoder(input_dim=input_dim, hidden_dim=hidden_dim)
    sae.load_state_dict(state)
    sae.eval()

    # Compute activations
    with torch.no_grad():
        activations = sae.encode(torch.from_numpy(embeddings)).numpy()

    ranked_features = rank_features_by_variance(activations)[: args.n_features]
    print(f"Naming {len(ranked_features)} features...")

    describer = CLIPDescriber()
    namer = LLMFeatureNamer(model=args.llm_model)
    feature_names: dict[str, str] = {}

    for fid in ranked_features:
        fi = get_top_images(activations, image_paths, fid, k=args.top_k)
        top_desc = describer.describe_images(fi.top_paths)
        bot_desc = describer.describe_images(fi.bottom_paths)
        name = namer.name_feature(top_desc, bot_desc)
        feature_names[str(fid)] = name
        print(f"  feature {fid:5d} -> {name!r}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(feature_names, indent=2))
    print(f"\nSaved feature names -> {args.output}")


if __name__ == "__main__":
    main()
