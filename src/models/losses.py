"""Loss functions for training the Sparse Autoencoder."""

import torch
import torch.nn.functional as F


def reconstruction_loss(x: torch.Tensor, x_hat: torch.Tensor) -> torch.Tensor:
    """Mean squared error between the original and reconstructed embeddings.

    Args:
        x: Original embeddings, shape ``(B, D)``.
        x_hat: Reconstructed embeddings, shape ``(B, D)``.

    Returns:
        Scalar MSE loss.
    """
    return F.mse_loss(x_hat, x)


def sparsity_loss(h: torch.Tensor) -> torch.Tensor:
    """L1 penalty on hidden activations to encourage sparsity.

    Args:
        h: Sparse activations, shape ``(B, hidden_dim)``.

    Returns:
        Scalar mean L1 norm across the batch.
    """
    return h.abs().mean()


