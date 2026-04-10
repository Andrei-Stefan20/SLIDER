"""Sparse Autoencoder (SAE) for disentangling DINOv2 feature space."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SparseAutoencoder(nn.Module):
    """Single-layer sparse autoencoder with ReLU activation.

    Architecture:
        encoder: Linear(input_dim -> hidden_dim) + ReLU
        decoder: Linear(hidden_dim -> input_dim)

    With ``tied_weights=True`` the decoder weight matrix is the transpose
    of the encoder weight matrix (no separate parameter).
    """

    def __init__(
        self,
        input_dim: int = 1024,
        hidden_dim: int = 8192,
        tied_weights: bool = False,
    ) -> None:
        """Initialise the SAE.

        Args:
            input_dim: Dimensionality of the input embeddings (1024 for
                DINOv2 ViT-L/14 CLS token).
            hidden_dim: Number of dictionary features / latent dimensions.
                Larger values give sparser, more interpretable features.
            tied_weights: If True, the decoder uses the transpose of the
                encoder weight matrix.  Reduces parameter count by ~50 %.
        """
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.tied_weights = tied_weights

        self.encoder = nn.Linear(input_dim, hidden_dim, bias=True)
        if not tied_weights:
            self.decoder = nn.Linear(hidden_dim, input_dim, bias=True)
        else:
            self.decoder_bias = nn.Parameter(torch.zeros(input_dim))

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialise encoder columns as unit vectors."""
        nn.init.kaiming_uniform_(self.encoder.weight)
        nn.init.zeros_(self.encoder.bias)
        if not self.tied_weights:
            nn.init.kaiming_uniform_(self.decoder.weight)
            nn.init.zeros_(self.decoder.bias)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Map input embeddings to sparse hidden activations.

        Args:
            x: Float32 tensor of shape ``(B, input_dim)``.

        Returns:
            Non-negative sparse activations of shape ``(B, hidden_dim)``.
        """
        return F.relu(self.encoder(x))

    def decode(self, h: torch.Tensor) -> torch.Tensor:
        """Reconstruct input from sparse activations.

        Args:
            h: Float32 tensor of shape ``(B, hidden_dim)``.

        Returns:
            Reconstructed embeddings of shape ``(B, input_dim)``.
        """
        if self.tied_weights:
            return h @ self.encoder.weight + self.decoder_bias
        return self.decoder(h)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Full encode–decode pass.

        Args:
            x: Float32 tensor of shape ``(B, input_dim)``.

        Returns:
            Tuple ``(x_hat, h)`` where:
            - ``x_hat``: reconstructed input, shape ``(B, input_dim)``
            - ``h``: sparse activations, shape ``(B, hidden_dim)``
        """
        h = self.encode(x)
        x_hat = self.decode(h)
        return x_hat, h
