"""CLI script for training a Sparse Autoencoder on pre-extracted embeddings.

Usage:
    python scripts/train_sae.py --embeddings data/processed/plantvillage_embeddings.npy \\
        --output models/ --config configs/plantvillage.yaml
"""

import argparse
from pathlib import Path

import yaml

from src.models.train_sae import train_sae


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a Sparse Autoencoder on DINOv2 embeddings."
    )
    parser.add_argument(
        "--embeddings",
        type=Path,
        required=True,
        help="Path to the .npy embeddings file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("models"),
        help="Directory to save checkpoints.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Optional YAML config file (overrides CLI defaults).",
    )
    parser.add_argument("--hidden-dim", type=int, default=8192)
    parser.add_argument("--lambda-sparsity", type=float, default=1e-3)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--tied-weights", action="store_true")
    args = parser.parse_args()

    # Merge config file values (config file wins over CLI defaults)
    cfg: dict = {}
    if args.config is not None:
        cfg = yaml.safe_load(args.config.read_text()).get("sae", {})

    hidden_dim = cfg.get("hidden_dim", args.hidden_dim)
    lambda_sparsity = cfg.get("lambda_sparsity", args.lambda_sparsity)
    lr = cfg.get("lr", args.lr)
    epochs = cfg.get("epochs", args.epochs)
    batch_size = cfg.get("batch_size", args.batch_size)

    print("Training SAE")
    print(f"  embeddings : {args.embeddings}")
    print(f"  hidden_dim : {hidden_dim}")
    print(f"  lambda_sp  : {lambda_sparsity}")
    print(f"  lr         : {lr}")
    print(f"  epochs     : {epochs}")
    print(f"  batch_size : {batch_size}")

    train_sae(
        embeddings_path=args.embeddings,
        output_dir=args.output,
        hidden_dim=hidden_dim,
        lambda_sparsity=lambda_sparsity,
        lr=lr,
        epochs=epochs,
        batch_size=batch_size,
        log_every=args.log_every,
        tied_weights=args.tied_weights,
    )


if __name__ == "__main__":
    main()
