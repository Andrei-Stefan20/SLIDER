"""Visual description of images using CLIP zero-shot classification."""

from pathlib import Path

import torch
from PIL import Image

from src.encoders.clip_encoder import CLIPEncoder

# Default vocabulary of visual attributes
DEFAULT_VOCAB: list[str] = [
    # Colour
    "bright", "dark", "colorful", "monochrome", "vivid", "pale",
    # Texture
    "smooth", "rough", "fuzzy", "glossy", "matte", "grainy",
    # Pattern
    "spotted", "striped", "uniform", "patterned", "irregular", "symmetric",
    # Shape
    "round", "elongated", "spiky", "flat", "curved", "angular",
    # Density
    "dense", "sparse", "clustered", "isolated",
    # Surface appearance
    "wrinkled", "wavy", "cracked", "intact", "decayed", "healthy",
]


class CLIPDescriber:
    """Generates visual descriptions for images via CLIP vocabulary matching.

    For each image the class computes cosine similarity between the image
    embedding and a set of text attribute embeddings, then returns the
    top-scoring words as a description.
    """

    def __init__(
        self,
        clip_encoder: CLIPEncoder | None = None,
        vocab: list[str] | None = None,
        top_n_words: int = 5,
    ) -> None:
        """Initialise the describer.

        Args:
            clip_encoder: Pre-instantiated :class:`CLIPEncoder`.  A new one
                is created if not provided.
            vocab: List of visual attribute words to rank.  Defaults to
                :data:`DEFAULT_VOCAB`.
            top_n_words: How many top words to include in each description.
        """
        self.encoder = clip_encoder if clip_encoder is not None else CLIPEncoder()
        self.vocab = vocab if vocab is not None else DEFAULT_VOCAB
        self.top_n_words = top_n_words

        self._text_embs: torch.Tensor = self.encoder.encode_text(
            [f"a {w} image" for w in self.vocab]
        )

    def describe_images(self, image_paths: list[Path | str]) -> list[str]:
        """Generate a short description for each image.

        Args:
            image_paths: Paths to image files.

        Returns:
            List of description strings, one per image.  Each description
            is a comma-separated list of the top-scoring vocabulary words.
        """
        descriptions: list[str] = []

        for path in image_paths:
            img = Image.open(path).convert("RGB")
            img_tensor = self.encoder.preprocess(img).unsqueeze(0)
            img_emb = self.encoder.encode_images(img_tensor).squeeze(0)

            sims = (self._text_embs @ img_emb).tolist()
            top_idx = sorted(range(len(sims)), key=lambda i: sims[i], reverse=True)
            top_words = [self.vocab[i] for i in top_idx[: self.top_n_words]]
            descriptions.append(", ".join(top_words))

        return descriptions
