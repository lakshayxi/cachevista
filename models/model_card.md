# Model Card — DriftMLP

## What it does
Binary classifier that detects conversation drift in Vision-Language Model sessions.
Given two consecutive CLIP embeddings (previous turn, current turn), it predicts
whether the conversation context has shifted (drift=1) or stayed consistent (drift=0).

## Architecture
- Input: 1536-dim vector (two 768-dim CLIP embeddings concatenated)
- Layers: Linear(1536→256) → ReLU → Dropout(0.3) → Linear(256→64) → ReLU → Linear(64→1) → Sigmoid
- Output: drift probability in [0, 1]. Threshold: 0.5

## Training data
- COCO val2017 subset (500 images)
- Synthetic conversation sequences: same-image augmentations = no drift, image switches = drift
- 80/20 train/test split

## Base encoder
CLIP ViT-L/14 (openai/clip-vit-large-patch14) — frozen, not fine-tuned.

## Intended use
Layer 3 of the CacheVista hybrid cache. Prevents stale cache hits when
the user's conversational focus shifts to a different aspect of the scene.
