from pathlib import Path

import numpy as np

from cachevista.config import load
from cachevista.mlp import train_model, save_model

cfg = load()
data_dir = Path(cfg["data"]["drift_data_dir"])

X_train = np.load(data_dir / "X_train.npy")
y_train = np.load(data_dir / "y_train.npy")
X_val = np.load(data_dir / "X_val.npy")
y_val = np.load(data_dir / "y_val.npy")

print(f"train: {len(X_train)} pairs  val: {len(X_val)} pairs  feature_dim: {X_train.shape[1]}")

model = train_model(
    X_train, y_train,
    X_val=X_val, y_val=y_val,
    epochs=cfg["training"]["epochs"],
    lr=cfg["training"]["lr"],
    batch_size=cfg["training"]["batch_size"],
)

out = Path(cfg["model"]["mlp_path"])
out.parent.mkdir(parents=True, exist_ok=True)
save_model(model, out)
print(f"saved to {out}")