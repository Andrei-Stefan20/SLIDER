# SLIDERS

Zero-shot interactive image retrieval using Sparse Autoencoders (SAEs) trained on DINOv2 features.

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

| Dataset | Description | Classes |
|---------|-------------|---------|
| **PlantVillage** | Plant leaf disease images | 38 disease/healthy classes |
| **Ceramics** | Archaeological ceramic shards | Typology-based categories |

Place raw images under `data/raw/<dataset>/` following the standard ImageFolder layout
(`data/raw/plantvillage/<class_name>/<image>.jpg`).

---

## Requirements

See `requirements.txt`. Key dependencies:

| Package | Purpose |
|---------|---------|
| `torch` / `torchvision` | Deep learning framework |
| `open-clip-torch` | CLIP ViT-L/14 for naming |
| `transformers` | DINOv2 utilities |
| `faiss-cpu` | Approximate nearest-neighbour search |
| `gradio` | Interactive web UI |
| `openai` | LLM-based feature naming (GPT-4o) |
