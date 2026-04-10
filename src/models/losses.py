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


def sae_loss(
    x: torch.Tensor,
    x_hat: torch.Tensor,
    h: torch.Tensor,
    lambda_sparsity: float = 1e-3,
) -> torch.Tensor:
    """Combined reconstruction + sparsity loss for SAE training.

    Loss = MSE(x, x_hat) + lambda_sparsity * L1(h)

    Args:
        x: Original embeddings, shape ``(B, D)``.
        x_hat: Reconstructed embeddings, shape ``(B, D)``.
        h: Sparse activations, shape ``(B, hidden_dim)``.
        lambda_sparsity: Weight of the L1 sparsity penalty.

    Returns:
        Scalar total loss.
    """
    rec = reconstruction_loss(x, x_hat)
    sparse = sparsity_loss(h)
    return rec + lambda_sparsity * sparse
