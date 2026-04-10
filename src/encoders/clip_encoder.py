"""CLIP ViT-L/14 encoder for image and text feature extraction."""

from typing import List

import numpy as np
import open_clip
import torch
import torch.nn.functional as F


class CLIPEncoder:
    """Wraps open-clip CLIP ViT-L/14 for image and text encoding.

    This encoder is used exclusively for feature naming and evaluation,
    *not* for retrieval (DINOv2 is used for that).
    """

    def __init__(self, model_name: str = "ViT-L-14", pretrained: str = "openai") -> None:
        """Load the CLIP model and its preprocessing transform.

        Args:
            model_name: open-clip model identifier.
            pretrained: Pretrained weights tag recognised by open-clip.
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained
        )
        self.tokenizer = open_clip.get_tokenizer(model_name)
        self.model.eval().to(self.device)

    @torch.no_grad()
    def encode_images(self, images: torch.Tensor) -> torch.Tensor:
        """Encode a batch of images into L2-normalised feature vectors.

        Args:
            images: Float32 tensor of shape ``(B, 3, H, W)``.

        Returns:
            L2-normalised float32 tensor of shape ``(B, 768)``.
        """
        images = images.to(self.device, dtype=torch.float32)
        features = self.model.encode_image(images)
        return F.normalize(features, dim=-1).cpu()

    @torch.no_grad()
    def encode_text(self, texts: List[str]) -> torch.Tensor:
        """Encode a list of text strings into L2-normalised feature vectors.

        Args:
            texts: List of strings to encode.

        Returns:
            L2-normalised float32 tensor of shape ``(N, 768)``.
        """
        tokens = self.tokenizer(texts).to(self.device)
        features = self.model.encode_text(tokens)
        return F.normalize(features, dim=-1).cpu()

    def similarity(self, image_emb: torch.Tensor, text_emb: torch.Tensor) -> float:
        """Compute cosine similarity between a single image and text embedding.

        Both inputs should already be L2-normalised (as returned by
        ``encode_images`` / ``encode_text``).

        Args:
            image_emb: Float32 tensor of shape ``(768,)`` or ``(1, 768)``.
            text_emb: Float32 tensor of shape ``(768,)`` or ``(1, 768)``.

        Returns:
            Scalar cosine similarity in ``[-1, 1]``.
        """
        image_emb = image_emb.flatten()
        text_emb = text_emb.flatten()
        return float(torch.dot(image_emb, text_emb).item())
