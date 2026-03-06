import sys
import numpy as np

from cachevista.config import load
from cachevista.mlp import train_model, save_model

cfg = load()
use_wandb = "--wandb" in sys.argv

X = np.load(cfg["data"]["drift_train_X"])
y = np.load(cfg["data"]["drift_train_y"])

print(f"training on {len(X)} pairs")
model = train_model(
    X, y,
    epochs=cfg["training"]["epochs"],
    lr=cfg["training"]["lr"],
    use_wandb=use_wandb,
)
save_model(model, cfg["model"]["mlp_path"])
print(f"saved to {cfg['model']['mlp_path']}")
