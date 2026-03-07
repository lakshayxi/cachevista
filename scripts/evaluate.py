from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, classification_report

from cachevista.config import load
from cachevista.mlp import load_model

cfg = load()
data_dir = Path(cfg["data"]["drift_data_dir"])

model = load_model(cfg["model"]["mlp_path"])
X_val = np.load(data_dir / "X_val.npy")
y_val = np.load(data_dir / "y_val.npy")

threshold = cfg["cache"]["drift_threshold"]
preds = [
    1 if model.predict_features(X_val[i]) > threshold else 0
    for i in range(len(X_val))
]

acc = accuracy_score(y_val, preds)
print(f"val accuracy: {acc:.4f}")
print(classification_report(y_val, preds, target_names=["no-drift", "drift"]))