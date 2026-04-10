"""Dataset classes for loading raw images and pre-extracted embeddings."""

from pathlib import Path

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import Dataset

# Standard ImageNet normalisation used by DINOv2
IMAGENET_TRANSFORM = T.Compose([
    T.Resize(256),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


class ImageFolderFlat(Dataset):
    """Recursively loads all images from a directory tree.

    Walks ``root`` recursively and collects every file whose extension
    matches :attr:`EXTENSIONS`.  Each ``__getitem__`` call returns a
    ``(tensor, path_str)`` pair so that downstream code can track which
    image corresponds to which embedding.

    Args:
        root: Root directory to search.
        transform: Torchvision transform applied to each PIL image.
            Defaults to :data:`IMAGENET_TRANSFORM`.
    """

    EXTENSIONS: frozenset[str] = frozenset({".jpg", ".jpeg", ".png", ".bmp", ".webp"})

    def __init__(
        self,
        root: Path | str,
        transform: T.Compose | None = None,
    ) -> None:
        self.root = Path(root)
        self.transform = transform or IMAGENET_TRANSFORM
        self.paths: list[Path] = sorted(
            p for p in self.root.rglob("*") if p.suffix.lower() in self.EXTENSIONS
        )

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, str]:
        """Return a preprocessed image tensor and its path string.

        Args:
            idx: Dataset index.

        Returns:
            Tuple ``(image_tensor, path_str)`` where ``image_tensor`` has
            shape ``(3, 224, 224)`` and ``path_str`` is the absolute path.
        """
        img = Image.open(self.paths[idx]).convert("RGB")
        return self.transform(img), str(self.paths[idx])


class EmbeddingDataset(Dataset):
    """Wraps a pre-extracted ``.npy`` embedding matrix as a PyTorch Dataset.

    Each ``__getitem__`` call returns a single float32 embedding vector.
    Useful for feeding embeddings to the SAE training loop without loading
    everything into memory at once (though the full array *is* memory-mapped).

    Args:
        npy_path: Path to a ``.npy`` file of shape ``(N, D)``.
        mmap: If True, open the file with ``numpy.memmap`` to avoid loading
            the full array into RAM.  Useful for very large datasets.
    """

    def __init__(self, npy_path: Path | str, mmap: bool = False) -> None:
        path = Path(npy_path)
        if mmap:
            self._data: np.ndarray = np.load(path, mmap_mode="r")
        else:
            self._data = np.load(path).astype(np.float32)

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, idx: int) -> torch.Tensor:
        """Return a single embedding as a float32 tensor.

        Args:
            idx: Dataset index.

        Returns:
            Float32 tensor of shape ``(D,)``.
        """
        return torch.from_numpy(self._data[idx].astype(np.float32))
