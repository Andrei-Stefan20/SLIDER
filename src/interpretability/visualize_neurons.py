"""Matplotlib visualisations for SAE feature analysis."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from src.naming.feature_namer import FeatureImages


def plot_feature_gallery(
    feature_images: FeatureImages,
    save_path: Path | str | None = None,
) -> None:
    """Show top-K (HIGH) and bottom-K (LOW) activating images side by side."""
    k = max(len(feature_images.top_paths), len(feature_images.bottom_paths))
    if k == 0:
        return

    fig, axes = plt.subplots(2, k, figsize=(2.5 * k, 5.5))
    if k == 1:
        axes = axes.reshape(2, 1)

    def _load(path: Path) -> Image.Image:
        return Image.open(path).convert("RGB").resize((112, 112))

    for col, (path, val) in enumerate(
        zip(feature_images.top_paths, feature_images.top_activations)
    ):
        axes[0, col].imshow(_load(path))
        axes[0, col].set_title(f"{val:.3f}", fontsize=8)
        axes[0, col].axis("off")

    for col, (path, val) in enumerate(
        zip(feature_images.bottom_paths, feature_images.bottom_activations)
    ):
        axes[1, col].imshow(_load(path))
        axes[1, col].set_title(f"{val:.3f}", fontsize=8)
        axes[1, col].axis("off")

    for col in range(len(feature_images.top_paths), k):
        axes[0, col].axis("off")
    for col in range(len(feature_images.bottom_paths), k):
        axes[1, col].axis("off")

    fig.text(0.01, 0.75, "HIGH", va="center", fontsize=11, fontweight="bold", color="#2a6ebb")
    fig.text(0.01, 0.25, "LOW",  va="center", fontsize=11, fontweight="bold", color="#bb2a2a")
    fig.suptitle(f"Feature {feature_images.feature_id}", fontsize=13, fontweight="bold")
    plt.tight_layout(rect=[0.04, 0, 1, 0.96])

    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


def plot_activation_histogram(
    activations: np.ndarray,
    feature_id: int,
    save_path: Path | str | None = None,
) -> None:
    """Distribution of non-zero activations for a single feature."""
    values = activations[:, feature_id]
    nonzero = values[values > 0]
    zero_frac = (values == 0).mean()

    fig, ax = plt.subplots(figsize=(7, 4))
    if len(nonzero) > 0:
        ax.hist(nonzero, bins=50, color="#2a6ebb", edgecolor="white", linewidth=0.4)
    ax.set_xlabel("Activation value (non-zero only)")
    ax.set_ylabel("Count")
    ax.set_title(
        f"Feature {feature_id} — activation distribution\n"
        f"({zero_frac*100:.1f}% inactive samples)"
    )
    plt.tight_layout()

    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


def plot_feature_variance_distribution(
    activations: np.ndarray,
    save_path: Path | str | None = None,
) -> None:
    """Histogram and sorted curve of per-feature activation variance."""
    variances = activations.var(axis=0)
    sorted_vars = np.sort(variances)[::-1]

    fig, axes = plt.subplots(1, 2, figsize=(13, 4))

    axes[0].hist(variances, bins=100, log=True, color="#5a9e6f", edgecolor="white", linewidth=0.3)
    axes[0].set_xlabel("Feature variance")
    axes[0].set_ylabel("Count (log scale)")
    axes[0].set_title("Distribution of feature variances")

    axes[1].plot(sorted_vars, color="#5a9e6f", linewidth=1.2)
    axes[1].set_xlabel("Feature rank (by variance)")
    axes[1].set_ylabel("Variance")
    axes[1].set_title("Feature variance (sorted descending)")
    axes[1].set_yscale("log")

    plt.tight_layout()

    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


def plot_dead_features(
    activations: np.ndarray,
    save_path: Path | str | None = None,
) -> None:
    """Dead vs alive feature pie chart and per-sample sparsity histogram."""
    active = (activations > 0).any(axis=0)
    dead_count = int((~active).sum())
    alive_count = int(active.sum())
    hidden_dim = activations.shape[1]

    sample_sparsity = (activations == 0).mean(axis=1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].pie(
        [dead_count, alive_count],
        labels=[f"Dead ({dead_count})", f"Alive ({alive_count})"],
        colors=["#e05c5c", "#5a9e6f"],
        autopct="%1.1f%%",
        startangle=90,
    )
    axes[0].set_title(f"Dead features out of {hidden_dim}")

    axes[1].hist(
        sample_sparsity, bins=50, color="#2a6ebb", edgecolor="white", linewidth=0.4
    )
    axes[1].set_xlabel("Fraction of zero activations per sample")
    axes[1].set_ylabel("Number of samples")
    axes[1].set_title(f"Per-sample sparsity  (mean={sample_sparsity.mean():.3f})")

    plt.tight_layout()

    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()
