# SETUP.md
## Environment Setup & How to Run

---

## Recommended: Google Colab

No local setup required. Open each notebook directly in Colab.

All dependencies are installed via `pip` at the top of each notebook.

**Run order:**
1. `notebooks/01_data_collection.ipynb`
2. `notebooks/02_retrieval_engines.ipynb`
3. `notebooks/03_experiment.ipynb`

> Each notebook saves intermediate outputs to the `data/` directory.  
> If using Colab, mount Google Drive or re-run notebooks in sequence each session.

---

## Local Setup (Optional)

### Requirements

- Python 3.9+
- pip

### Install dependencies

```bash
pip install -r requirements.txt
```

### `requirements.txt`

```
wikipedia-api==0.6.0
rank-bm25==0.2.2
sentence-transformers==2.7.0
faiss-cpu==1.7.4
numpy==1.26.4
matplotlib==3.8.4
pandas==2.2.2
jupyter==1.0.0
tqdm==4.66.4
```

### Run notebooks locally

```bash
jupyter notebook
```

Then open notebooks in order from the `notebooks/` directory.

---

## Key Library Notes

### `rank_bm25`
Pure Python BM25 implementation. No GPU required.  
GitHub: https://github.com/dorianbrown/rank_bm25

### `sentence-transformers`
Provides pre-trained embedding models.  
Model used: `all-MiniLM-L6-v2` (22M parameters, 384-dim, Apache 2.0 license)  
Auto-downloads on first use (~90MB).  
Docs: https://www.sbert.net

### `faiss-cpu`
Facebook AI Similarity Search — vector index library.  
CPU version sufficient for our dataset size (~100 chunks).  
GitHub: https://github.com/facebookresearch/faiss

### `wikipedia-api`
Clean Python wrapper for the Wikipedia API.  
GitHub: https://github.com/martin-majlis/Wikipedia-API

---

## Data Collection Notes

The data collection notebook (`01_data_collection.ipynb`) fetches Wikipedia articles via the Wikipedia API. An internet connection is required for this step.

If Wikipedia is unavailable, a pre-collected `data/chunks.json` is included in the repository and can be used directly in notebooks 02 and 03.

---

## Reproducibility

All random seeds are fixed where applicable. The retrieval results are deterministic:
- BM25: deterministic (no randomness)
- FAISS IndexFlatIP: exact search (no approximation), fully deterministic
- RRF: deterministic given the two input rankings

Running the notebooks in order on the same `chunks.json` will produce identical results.
