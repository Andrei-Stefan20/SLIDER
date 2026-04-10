"""CLI script for extracting DINOv2 embeddings from a dataset.

Usage:
    python scripts/extract_embeddings.py --dataset plantvillage --output data/processed/
"""

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.loader import ImageFolderFlat
from src.encoders.dino_encoder import DINOEncoder
from src.utils.io import save_embeddings, save_image_paths


def collate_fn(batch):
    images, paths = zip(*batch)
    return torch.stack(images), list(paths)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract DINOv2 embeddings from an image dataset."
    )
    parser.add_argument(
        "--dataset",
        required=True,
        help="Dataset name (used for output filenames, e.g. plantvillage).",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="Path to raw dataset root (defaults to data/raw/<dataset>).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed"),
        help="Directory where embeddings and image paths are saved.",
    )
    parser.add_argument(
        "--batch-size", type=int, default=64, help="Inference batch size."
    )
    parser.add_argument(
        "--use-patches",
        action="store_true",
        help="Extract patch tokens instead of CLS token.",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        if args.batch_size == 64:  # only override if still at default
            args.batch_size = 32
            print("Non-CUDA device detected — batch size reduced to 32")

    input_dir = args.input or Path("data/raw") / args.dataset
    output_dir = args.output
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading images from: {input_dir}")
    dataset = ImageFolderFlat(input_dir)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn,
    )
    print(f"  Found {len(dataset)} images.")

    encoder = DINOEncoder(use_patches=args.use_patches)

    all_embeddings: list[np.ndarray] = []
    all_paths: list[str] = []

    for images, paths in tqdm(loader, desc="Extracting embeddings"):
        emb = encoder.encode(images).numpy()
        all_embeddings.append(emb)
        all_paths.extend(paths)

    embeddings = np.concatenate(all_embeddings, axis=0).astype(np.float32)

    emb_path = output_dir / f"{args.dataset}_embeddings.npy"
    paths_path = output_dir / f"{args.dataset}_image_paths.json"

    save_embeddings(embeddings, emb_path)
    save_image_paths(all_paths, paths_path)

    print(f"Saved embeddings ({embeddings.shape}) -> {emb_path}")
    print(f"Saved image paths ({len(all_paths)})  -> {paths_path}")


if __name__ == "__main__":
    main()
