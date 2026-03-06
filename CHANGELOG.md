# Changelog

## [Unreleased]

### Added
- Layer 1: MD5 hash exact cache
- Layer 2: FAISS semantic similarity cache (cosine, threshold=0.90)
- Layer 3: MLP drift predictor (1536→256→64→1)
- CLIP ViT-L/14 encoder with MPS support
- Config system via configs/config.yaml
- COCO val2017 download script (500 images)
- Drift training data generator with 80/20 train/test split
- scripts/train.py and scripts/evaluate.py for reproducibility
- pytest test suite (12 tests across core, encoder, FAISS, MLP)
- wandb experiment tracking (optional, --wandb flag)
- GitHub Actions CI on push
- ruff linting config
- Model card, data README, contributing guide

### Fixed
- NumPy 2.x vs PyTorch compatibility (pinned numpy<2)
- CLIPProcessor → CLIPImageProcessor for image-only encoding
- Hardcoded paths replaced with config system
- Bloated environment.yml replaced with clean requirements.txt
- data/ and models/ excluded from git via .gitignore
