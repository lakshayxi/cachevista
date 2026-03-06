# CacheVista

Multimodal memory caching for Vision-Language Models. Reduces redundant image encoding by caching CLIP embeddings across conversation turns using a 3-layer hybrid cache.

## How it works

| Layer | Method | Handles |
|-------|--------|---------|
| L1 | MD5 hash | Exact duplicate images |
| L2 | FAISS cosine similarity | Near-duplicate images |
| L3 | MLP drift predictor | Conversation context shifts |

## Setup
```bash
conda create -n cachevista python=3.11
conda activate cachevista
pip install -r requirements.txt
```

## Reproduce
```bash
# 1. Download COCO subset
python -m cachevista.download_coco

# 2. Generate drift training pairs
python -m cachevista.generate_drift_data

# 3. Train the MLP
python scripts/train.py

# 4. Evaluate
python scripts/evaluate.py

# 5. Run tests
pytest cachevista/tests/
```

## Project structure
```
cachevista/
    cachevista/        # core package
        core.py        # cache strategies (L1, L2, L3)
        encoder.py     # CLIP ViT-L/14 wrapper
        mlp.py         # drift predictor MLP
        config.py      # config loader
        tests/         # pytest test suite
    configs/
        config.yaml    # all hyperparameters and paths
    scripts/
        train.py       # train the MLP
        evaluate.py    # evaluate on held-out test set
    data/
        README.md      # how to reproduce the dataset
    models/
        model_card.md  # model documentation
```

## Results

Coming after benchmark run.

## Device

Developed on MacBook Air M4 (MPS backend). Runs on CPU too.
