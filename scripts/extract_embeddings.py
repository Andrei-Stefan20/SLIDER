"""CLI script for extracting DINOv2 embeddings from a dataset.

Usage:
    python scripts/extract_embeddings.py --dataset plantvillage --output data/processed/
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T
from tqdm import tqdm

from src.encoders.dino_encoder import DINOEncoder


IMAGENET_TRANSFORM = T.Compose([
    T.Resize(256),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


class ImageFolderFlat(Dataset):
    """Recursively loads all images from a directory tree.

    Args:
        root: Root directory to search.
        extensions: Accepted file extensions.
    """

    EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    def __init__(self, root: Path, transform=None) -> None:
        self.paths = sorted(
            p for p in root.rglob("*") if p.suffix.lower() in self.EXTENSIONS
        )
        self.transform = transform or IMAGENET_TRANSFORM

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, str]:
        img = Image.open(self.paths[idx]).convert("RGB")
        return self.transform(img), str(self.paths[idx])


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

    np.save(emb_path, embeddings)
    paths_path.write_text(json.dumps(all_paths, indent=2))

    print(f"Saved embeddings ({embeddings.shape}) -> {emb_path}")
    print(f"Saved image paths ({len(all_paths)})  -> {paths_path}")


if __name__ == "__main__":
    main()
