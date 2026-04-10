# SLIDERS

## What is this?

You upload an image. The system finds visually similar images in a dataset — but unlike standard retrieval, it also shows you a set of sliders, each labelled with a visual concept like "leaf margin complexity" or "surface texture uniformity". These concepts are discovered automatically: a Sparse Autoencoder (SAE) is trained on DINOv2 embeddings of the dataset, and its learned features are named by showing top/bottom activating images to CLIP and GPT-4o. When you move a slider, the query embedding is shifted along the corresponding SAE feature direction, steering retrieval toward images that have more or less of that property. No text input is required at any point — the sliders are the interface.

---

## Architecture

```
Raw Images
    │
    ▼
┌─────────────────┐
│  DINOv2 ViT-L/14│  ← CLS token (1024-dim)
└────────┬────────┘
         │ embeddings.npy
         ▼
┌─────────────────────────────────┐
│  Sparse Autoencoder (SAE)       │
│  Linear → ReLU → Linear        │
│  1024 → 8192 → 1024            │
└────────┬──────────────┬─────────┘
         │ h (sparse)   │ x_hat
         │              │
         ▼              ▼ (reconstruction loss + L1)
  Feature directions   Training
  (decoder columns)
         │
         ▼
┌──────────────────────────────────┐
│  CLIP ViT-L/14                   │
│  + LLM (GPT-4o)                  │
│  → Named sliders                 │
│  e.g. "leaf margin complexity"   │
└─────────────┬────────────────────┘
              │
              ▼
┌─────────────────────────────────────────────────┐
│  Gradio UI                                      │
│  ┌──────────┐  ┌──────────────────────────────┐ │
│  │  Query   │  │  Sliders (named SAE features) │ │
│  │  Image   │  │  ○────────────●──────────○   │ │
│  └────┬─────┘  └──────────────┬───────────────┘ │
│       │    Steered query emb  │                  │
│       └──────────┬────────────┘                  │
│                  ▼                               │
│          FAISS IndexFlatIP                       │
│          (cosine similarity)                     │
│                  │                               │
│                  ▼                               │
│          Retrieved Gallery                       │
└─────────────────────────────────────────────────┘
```

## How it works

DINOv2 embeddings are extracted once and stored on disk. The SAE is trained on these embeddings to decompose them into sparse, roughly monosemantic features — each decoder column becomes a direction in embedding space that corresponds to some visual property. To name these directions, we find the images that activate each feature most and least, describe them with CLIP against a vocabulary of visual adjectives, and pass those descriptions to GPT-4o to generate a short label. At retrieval time, the query embedding is shifted by a weighted sum of selected feature directions before the FAISS search, producing results that reflect the slider adjustments.

---

## Installation

```bash
git clone <repo-url>
cd SLIDER
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Set your OpenAI API key (required for the naming step only):

```bash
export OPENAI_API_KEY=sk-...
```

---

## Usage — 5-step pipeline

### Step 1 — Extract DINOv2 embeddings

```bash
python scripts/extract_embeddings.py \
    --dataset plantvillage \
    --input data/raw/plantvillage \
    --output data/processed/ \
    --batch-size 64
```

Outputs: `data/processed/plantvillage_embeddings.npy`, `plantvillage_image_paths.json`

### Step 2 — Train the Sparse Autoencoder

```bash
python scripts/train_sae.py \
    --embeddings data/processed/plantvillage_embeddings.npy \
    --output models/ \
    --config configs/plantvillage.yaml
```

Outputs: `models/sae_best.pt`

### Step 3 — Name SAE features

```bash
python scripts/name_features.py \
    --embeddings data/processed/plantvillage_embeddings.npy \
    --image-paths data/processed/plantvillage_image_paths.json \
    --sae-model models/sae_best.pt \
    --output models/feature_names.json \
    --n-features 20
```

Outputs: `models/feature_names.json`

### Step 4 — Build the retrieval index

```bash
python scripts/build_index.py \
    --embeddings data/processed/plantvillage_embeddings.npy \
    --output data/processed/index.faiss
```

Outputs: `data/processed/index.faiss`

### Step 5 — Launch the interactive UI

```bash
python src/ui/app.py
```

Open `http://127.0.0.1:7860` in your browser.

---

## Evaluation

```bash
python scripts/evaluate.py \
    --embeddings data/processed/plantvillage_embeddings.npy \
    --image-paths data/processed/plantvillage_image_paths.json \
    --index data/processed/index.faiss \
    --sae-model models/sae_best.pt \
    --feature-names models/feature_names.json
```

Reports **Recall@1/5/10** (same-class ground truth) and **CLIP alignment** for each named slider.

---

## Datasets

| Dataset | Description | Classes | Notes |
|---------|-------------|---------|-------|
| **PlantVillage** | Plant leaf disease images | 38 disease/healthy classes | Main benchmark — disease features are visually continuous, ideal for slider discovery |
| **Ceramics** | Archaeological ceramic shards | Typology-based categories | Stress test — iconographic features push the limits of the backbone |

Place raw images under `data/raw/<dataset>/` following the standard ImageFolder layout
(`data/raw/plantvillage/<class_name>/<image>.jpg`).

---

## Known limitations

- SAE features may not be interpretable on all datasets. Sparsity and hidden dimension interact with the visual diversity of the corpus; expect dead or uninterpretable features, especially on small or iconographically constrained collections.
- Naming quality depends on CLIP vocabulary coverage and the LLM prompt. The pipeline can produce plausible-sounding but inaccurate names when the true feature captures something outside the default adjective vocabulary.
- Retrieval is zero-shot; no supervised fine-tuning is performed. Recall numbers should be read as a measure of the backbone's representational quality for a given dataset, not as a trained retrieval system.

---

## Requirements

See `requirements.txt`. Key dependencies:

| Package | Purpose |
|---------|---------|
| `torch` / `torchvision` | Deep learning framework |
| `open-clip-torch` | CLIP ViT-L/14 for naming |
| `transformers` | DINOv2 utilities |
| `faiss-cpu` | Nearest-neighbour search |
| `gradio` | Interactive web UI |
| `openai` | LLM-based feature naming (GPT-4o) |
