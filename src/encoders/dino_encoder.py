"""DINOv2 ViT-L/14 encoder for extracting image features."""

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm


class DINOEncoder:
    """Wraps DINOv2 ViT-L/14 for CLS-token and patch-token extraction.

    Loads the model from ``facebookresearch/dinov2`` via torch.hub.
    All inference is performed with ``torch.no_grad()`` in float32.
    """

    def __init__(self, use_patches: bool = False) -> None:
        """Initialise and load the DINOv2 ViT-L/14 model.

        Args:
            use_patches: If True, return concatenated patch tokens instead
                of the CLS token.  CLS token has shape ``(B, 1024)``;
                patch tokens have shape ``(B, N_patches, 1024)``.
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.use_patches = use_patches

        self.model: nn.Module = torch.hub.load(
            "facebookresearch/dinov2", "dinov2_vitl14"
        )
        self.model.eval().to(self.device)

    @torch.no_grad()
    def encode(self, images: torch.Tensor) -> torch.Tensor:
        """Encode a batch of images.

        Args:
            images: Float32 tensor of shape ``(B, 3, H, W)``, values in
                ``[0, 1]`` or pre-normalised.

        Returns:
            Float32 tensor.  Shape ``(B, 1024)`` when ``use_patches=False``,
            or ``(B, N_patches, 1024)`` when ``use_patches=True``.
        """
        images = images.to(self.device, dtype=torch.float32)

        if self.use_patches:
            out = self.model.forward_features(images)
            return out["x_norm_patchtokens"].cpu()
        else:
            return self.model(images).cpu()

    @torch.no_grad()
    def encode_dataset(
        self,
        dataloader: DataLoader,
        save_path: Path | None = None,
    ) -> np.ndarray:
        """Encode an entire dataset and optionally persist to disk.

        Args:
            dataloader: Yields ``(images, *_)`` batches where ``images`` is a
                float tensor of shape ``(B, 3, H, W)``.
            save_path: If provided, save the resulting array as a ``.npy``
                file at this path.

        Returns:
            Float32 NumPy array of shape ``(N, 1024)`` (CLS-token mode) or
            ``(N, N_patches, 1024)`` (patch mode).
        """
        all_embeddings: list[np.ndarray] = []

        for batch in tqdm(dataloader, desc="Encoding dataset"):
            images = batch[0] if isinstance(batch, (list, tuple)) else batch
            embeddings = self.encode(images)
            all_embeddings.append(embeddings.numpy())

        result = np.concatenate(all_embeddings, axis=0).astype(np.float32)

        if save_path is not None:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(save_path, result)

        return result
