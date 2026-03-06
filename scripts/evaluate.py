import numpy as np
from sklearn.metrics import accuracy_score, classification_report

from cachevista.config import load
from cachevista.mlp import load_model

cfg = load()

model = load_model(cfg["model"]["mlp_path"])

X_test = np.load(cfg["data"]["drift_test_X"])
y_test = np.load(cfg["data"]["drift_test_y"])

preds = [1 if model.predict(X_test[i, :768], X_test[i, 768:]) > cfg["cache"]["drift_threshold"] else 0 for i in range(len(X_test))]

acc = accuracy_score(y_test, preds)
print(f"test accuracy: {acc:.4f}")
print(classification_report(y_test, preds, target_names=["no-drift", "drift"]))
