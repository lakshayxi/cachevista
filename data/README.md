# Data

This directory is not committed to git.

## To reproduce

### Test images (cat + dog)
Already downloaded by the setup script. If missing:
```bash
python -m cachevista.download_coco
```

### COCO val2017 subset (500 images)
```bash
python -m cachevista.download_coco
```
Downloads ~500 images from COCO val2017 to `data/coco/images/`.

### Drift training pairs
Generated from COCO images using CLIP embeddings:
```bash
python -m cachevista.generate_drift_data
```
Produces `drift_X_coco.npy`, `drift_y_coco.npy`, `drift_X_test.npy`, `drift_y_test.npy`.
