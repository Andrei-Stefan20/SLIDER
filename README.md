# SLIDERS

Interactive image retrieval steered by interpretable visual features.

Upload a query image and adjust named sliders — e.g. *leaf margin complexity* or *surface texture uniformity* — to pull results toward images that have more or less of each property. No text input required at any point.

The sliders correspond to directions in DINOv2 embedding space discovered by a Sparse Autoencoder (SAE). Each direction is automatically named by showing its most- and least-activating images to CLIP and GPT-4o.

---

## How it works

```
Raw Images
    │
    ▼
DINOv2 ViT-L/14  →  embeddings.npy  (1024-dim CLS token per image)
    │
    ▼
Sparse Autoencoder  →  sae_best.pt  (1024 → 8192 → 1024)
    │
    ├── decoder columns  →  feature directions in embedding space
    │
    └── CLIP + GPT-4o  →  feature_names.json  (e.g. "leaf margin complexity")
    │
    ▼
FAISS IndexFlatIP  →  index.faiss  (cosine similarity search)
    │
    ▼
Gradio UI  —  upload image, adjust sliders, browse steered results
```

At query time: `q' = q + Σ(alpha_i · direction_i)`, then L2-normalise and search.

---

## Requirements

- Python 3.10+
- ~4 GB disk space for model weights (DINOv2 + CLIP download on first run)
- OpenAI API key — **only for Step 3** (feature naming); the rest runs offline

---

## Installation

```bash
git clone <repo-url>
cd SLIDER
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Set your OpenAI key if you plan to run Step 3:

```bash
cp .env.example .env
# edit .env and fill in OPENAI_API_KEY
export $(cat .env | xargs)   # or set the variable however you prefer
```

---

## Usage

Place raw images under `data/raw/<dataset>/` in standard ImageFolder layout:

```
data/raw/plantvillage/
    Apple___Black_rot/
        img001.jpg
        img002.jpg
    Apple___healthy/
        img003.jpg
    ...
```

### Step 1 — Extract DINOv2 embeddings

```bash
python scripts/extract_embeddings.py \
    --dataset plantvillage \
    --input data/raw/plantvillage \
    --output data/processed/
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

### Step 3 — Name SAE features  *(requires OpenAI API key)*

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

### Step 5 — Launch the UI

```bash
python src/ui/app.py
```

Open `http://127.0.0.1:7860`.  
The UI reads from the default paths in Step 1–4. If `feature_names.json` is missing, sliders fall back to the top-N features by activation variance.

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

Reports **Recall@1/5/10** (same-class ground truth) and **CLIP alignment** per named slider.

---

## Datasets

| Dataset | Description | Classes |
|---------|-------------|---------|
| **PlantVillage** | Plant leaf disease images | 38 |
| **Ceramics** | Archaeological ceramic shards | typology-based |

Config files for both are in `configs/`. Key SAE hyperparameters differ between datasets (see the YAML files).

---

## Project structure

```
configs/          YAML configs for each dataset
scripts/          CLI entry points (run in order: extract → train → name → index)
src/
  data/           Image and embedding dataset classes
  encoders/       DINOv2 and CLIP wrappers
  models/         SAE architecture and training loop
  retrieval/      FAISS index, query, and steering logic
  naming/         CLIP describer + LLM feature namer
  evaluation/     Recall@K and CLIP alignment metrics
  interpretability/ Activation analysis and Matplotlib visualisations
  ui/             Gradio app
  utils/          Logging, serialisation, device detection
notebooks/        Exploratory analysis (dataset structure, SAE features)
```

---

## Dependencies

| Package | Purpose |
|---------|---------|
| `torch` / `torchvision` | Deep learning |
| `open-clip-torch` | CLIP ViT-L/14 for feature naming and evaluation |
| `faiss-cpu` | Nearest-neighbour search |
| `gradio` | Interactive web UI |
| `openai` | GPT-4o for feature naming (Step 3 only) |
| `matplotlib` | Visualisation notebooks and interpretability plots |
| `tqdm` | Progress bars |
| `pyyaml` | Config parsing |
| `pillow` | Image loading |

---

## Known limitations

- SAE features may not be interpretable on all datasets. Dead or uninterpretable features are common on small or iconographically constrained collections.
- Naming quality depends on CLIP vocabulary coverage and the LLM prompt. Names can be plausible-sounding but inaccurate when the feature captures something outside the default adjective vocabulary.
- Retrieval is zero-shot; no supervised fine-tuning is performed. Recall numbers measure backbone quality for a given dataset, not a trained retrieval system.
- DINOv2 inference falls back to CPU on Apple Silicon (MPS) due to known compatibility issues; this makes embedding extraction slow on Mac.
