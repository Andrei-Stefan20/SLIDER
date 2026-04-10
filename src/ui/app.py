"""Gradio-based interactive retrieval UI for SLIDERS."""

import argparse
import json
from pathlib import Path

import gradio as gr
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image

from src.encoders.dino_encoder import DINOEncoder
from src.models.sae import SparseAutoencoder
from src.naming.feature_namer import rank_features_by_variance
from src.retrieval.index import load_index
from src.retrieval.query import search_with_sliders
from src.utils.io import normalize_embeddings
from src.utils.logging import get_logger

logger = get_logger(__name__)

DEFAULT_INDEX_PATH = Path("data/processed/index.faiss")
DEFAULT_SAE_PATH = Path("models/sae_best.pt")
DEFAULT_EMBEDDINGS_PATH = Path("data/processed/embeddings.npy")
DEFAULT_IMAGE_PATHS_JSON = Path("data/processed/image_paths.json")
N_SLIDERS = 10
RETRIEVAL_K = 20

_DINO_TRANSFORM = T.Compose([
    T.Resize(224),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def load_resources(
    index_path: Path = DEFAULT_INDEX_PATH,
    sae_path: Path = DEFAULT_SAE_PATH,
    embeddings_path: Path = DEFAULT_EMBEDDINGS_PATH,
    image_paths_json: Path = DEFAULT_IMAGE_PATHS_JSON,
    dataset: str | None = None,
) -> tuple:
    """Load all pre-built artefacts needed for interactive retrieval.

    Args:
        index_path: Path to the FAISS index file.
        sae_path: Path to the SAE checkpoint.
        embeddings_path: Path to the embeddings .npy file.
        image_paths_json: Path to the image paths JSON file.
        dataset: If provided, override ``embeddings_path`` and
            ``image_paths_json`` with ``data/processed/{dataset}_*.{ext}``
            paths.  ``index_path`` and ``sae_path`` are not affected.

    Returns:
        Tuple of ``(dino, sae, index, embeddings, image_paths, feature_ids,
        feature_names)``.
    """
    if dataset is not None:
        embeddings_path = Path(f"data/processed/{dataset}_embeddings.npy")
        image_paths_json = Path(f"data/processed/{dataset}_image_paths.json")

    dino = DINOEncoder(use_patches=False)

    sae_state = torch.load(sae_path, map_location="cpu")
    input_dim = sae_state["encoder.weight"].shape[1]
    hidden_dim = sae_state["encoder.weight"].shape[0]
    sae = SparseAutoencoder(input_dim=input_dim, hidden_dim=hidden_dim)
    sae.load_state_dict(sae_state)
    sae.eval()

    index = load_index(index_path)
    embeddings = np.load(embeddings_path)
    image_paths = json.loads(image_paths_json.read_text())

    names_path = sae_path.parent / "feature_names.json"
    if names_path.exists():
        all_names = json.loads(names_path.read_text())
        feature_ids = [int(k) for k in list(all_names.keys())[:N_SLIDERS]]
        feature_names = [all_names[str(fid)] for fid in feature_ids]
    else:
        with torch.no_grad():
            activations = sae.encode(
                torch.from_numpy(embeddings.astype(np.float32))
            ).numpy()
        ranked = rank_features_by_variance(activations)
        feature_ids = ranked[:N_SLIDERS]
        feature_names = [f"Feature {fid}" for fid in feature_ids]
        del activations

    return dino, sae, index, embeddings, image_paths, feature_ids, feature_names


def build_app(
    index_path: Path = DEFAULT_INDEX_PATH,
    sae_path: Path = DEFAULT_SAE_PATH,
    embeddings_path: Path = DEFAULT_EMBEDDINGS_PATH,
    image_paths_json: Path = DEFAULT_IMAGE_PATHS_JSON,
    dataset: str | None = None,
) -> gr.Blocks:
    """Construct and return the Gradio Blocks app.

    Args:
        index_path: Path to the FAISS index file.
        sae_path: Path to the SAE checkpoint.
        embeddings_path: Path to the embeddings .npy file.
        image_paths_json: Path to the image paths JSON file.
        dataset: Passed through to :func:`load_resources` for path resolution.

    Returns:
        Configured :class:`gr.Blocks` application.
    """
    dino, sae, index, embeddings, image_paths, feature_ids, feature_names = (
        load_resources(index_path, sae_path, embeddings_path, image_paths_json, dataset)
    )

    def retrieve(query_image: np.ndarray | None, *slider_values: float):
        if query_image is None:
            return []

        pil_img = Image.fromarray(query_image).convert("RGB")
        img_tensor = _DINO_TRANSFORM(pil_img).unsqueeze(0)
        query_emb = dino.encode(img_tensor).squeeze(0).numpy()
        query_emb = normalize_embeddings(query_emb.reshape(1, -1)).squeeze(0)

        slider_config = {
            fid: float(alpha)
            for fid, alpha in zip(feature_ids, slider_values)
            if abs(alpha) > 1e-6
        }

        _, indices = search_with_sliders(
            index, query_emb, sae, slider_config, k=RETRIEVAL_K
        )

        result_images = []
        for idx in indices:
            if 0 <= idx < len(image_paths):
                try:
                    img = Image.open(image_paths[idx]).convert("RGB")
                    result_images.append(img)
                except Exception as e:
                    logger.warning(f"Could not load image {image_paths[idx]}: {e}")
        return result_images

    with gr.Blocks(title="SLIDERS — Interactive Image Retrieval") as demo:
        gr.Markdown("# SLIDERS\nZero-shot interactive image retrieval via SAE feature steering.")

        with gr.Row():
            query_input = gr.Image(label="Query Image", type="numpy", height=300)

        gr.Markdown("### Sliders")
        slider_components: list[gr.Slider] = []
        for name in feature_names:
            s = gr.Slider(
                minimum=-3.0,
                maximum=3.0,
                value=0.0,
                step=0.1,
                label=name,
            )
            slider_components.append(s)

        retrieve_btn = gr.Button("Retrieve", variant="primary")
        gallery = gr.Gallery(
            label="Retrieved Images",
            columns=5,
            height=400,
            object_fit="cover",
        )

        retrieve_btn.click(
            fn=retrieve,
            inputs=[query_input] + slider_components,
            outputs=gallery,
        )

    return demo


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Launch the SLIDERS Gradio UI.")
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Dataset name (e.g. plantvillage). Resolves embeddings and image-paths "
             "from data/processed/<dataset>_*.{npy,json}.",
    )
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX_PATH)
    parser.add_argument("--sae-model", type=Path, default=DEFAULT_SAE_PATH)
    parser.add_argument("--share", action="store_true", help="Create a public Gradio link.")
    args = parser.parse_args()

    app = build_app(
        index_path=args.index,
        sae_path=args.sae_model,
        dataset=args.dataset,
    )
    app.launch(share=args.share)
