import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from cachevista.utils import get_device as _get_device


def _build_features(emb1: np.ndarray, emb2: np.ndarray) -> np.ndarray:
    """
    Explicit similarity features from two joint embeddings.
    Works for any embedding dimension — image-only (768) or joint (1152).

      |emb1 - emb2|  : per-dim absolute difference (1152-dim)
      emb1 * emb2    : element-wise product (1152-dim)
      cosine_sim     : scalar dot product — valid only for unit-norm inputs (1-dim)

    Output dim = 2*input_dim + 1  (2305 for 1152-dim joint embeddings)

    Design note: diff, prod, and cosine are correlated for unit-norm inputs
    (cosine = sum(prod), ||diff||^2 = 2 - 2*cosine). They encode the same
    pairwise geometry in different forms. Richer representation can still help
    the MLP learn non-linear interactions, but a simpler feature set has not
    been ablated.

    Symmetry note: abs(diff) makes _build_features(a, b) == _build_features(b, a)
    by construction. Drift prediction is therefore symmetric — the model cannot
    distinguish "query drifted from candidate" vs "candidate drifted from query".
    This is a deliberate simplification. If asymmetric drift matters, replace
    abs(diff) with the signed difference (emb1 - emb2).

    Both inputs must be unit-norm float32 vectors of the same shape.
    """
    if abs(np.linalg.norm(emb1) - 1.0) > 1e-4:
        raise ValueError(f"emb1 must be unit-norm, got norm={np.linalg.norm(emb1):.6f}")
    if abs(np.linalg.norm(emb2) - 1.0) > 1e-4:
        raise ValueError(f"emb2 must be unit-norm, got norm={np.linalg.norm(emb2):.6f}")

    diff = np.abs(emb1 - emb2)
    prod = emb1 * emb2
    cosine = np.array([np.dot(emb1, emb2)], dtype=np.float32)
    return np.concatenate([diff, prod, cosine])


class DriftMLP(nn.Module):
    """
    Binary drift classifier. Input: feature vector from _build_features.
    Output: drift probability in [0, 1]. High score = drift = don't reuse cache.

    Architecture: Linear(input_dim→256) → ReLU → Dropout(p) →
                  Linear(256→64) → ReLU → Dropout(p) → Linear(64→1)

    Config note: dropout rate and drift_threshold (0.5) are passed explicitly
    at construction / inference time rather than read from config.yaml, so
    threshold tuning does not require a code change — just pass a different
    threshold to the caller.
    """

    def __init__(self, input_dim: int = 2305, dropout: float = 0.2):
        super().__init__()
        self.input_dim = input_dim
        self.dropout = dropout
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(dropout),  # symmetric regularization on both hidden layers
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)  # squeeze last dim — safe for any batch size

    def predict(self, emb_query: np.ndarray, emb_candidate: np.ndarray) -> float:
        """
        Returns drift probability in [0, 1] for (query_joint_emb, candidate_joint_emb).
        Both embeddings must be unit-norm joint vectors (enforced by _build_features).
        Enforces eval mode for the duration of the call.

        The drift decision threshold (default 0.5) lives in the caller — core.py does
        `if mlp.predict(emb, candidate) > threshold` so it can be tuned via config
        without changing this function.
        """
        if emb_query.shape != emb_candidate.shape:
            raise ValueError(
                f"shape mismatch: {emb_query.shape} vs {emb_candidate.shape}"
            )
        return self.predict_features(_build_features(emb_query, emb_candidate))

    def predict_features(self, features: np.ndarray) -> float:
        """
        Takes a pre-built feature vector and returns drift probability.
        Enforces eval mode for the duration of the call — safe to call even
        if the model is in train() mode (e.g. during training monitoring).
        """
        was_training = self.training
        self.eval()
        try:
            device = next(self.parameters()).device
            x = torch.from_numpy(features).unsqueeze(0).to(device)
            with torch.inference_mode():
                logit = self.forward(x)
            return float(torch.sigmoid(logit))
        finally:
            if was_training:
                self.train()


def train_model(X_train, y_train, X_val=None, y_val=None,
                epochs=50, lr=1e-3, batch_size=64, dropout=0.2,
                pos_weight=None, seed=42, use_wandb=False):
    """
    Train the DriftMLP.

    Args:
        pos_weight: passed to BCEWithLogitsLoss to correct class imbalance.
                    With the default 1:2 positive:negative ratio from
                    generate_drift_data.py, pass pos_weight=2.0.
        seed:       fixes torch, DataLoader shuffle, and internal val split
                    for full reproducibility.
        dropout:    applied after both hidden layers (symmetric).
    """
    if use_wandb:
        import wandb
        wandb.init(project="cachevista", config={
            "epochs": epochs, "lr": lr, "batch_size": batch_size,
            "dropout": dropout, "pos_weight": pos_weight, "seed": seed,
        })

    torch.manual_seed(seed)

    device = _get_device()
    print(f"training on {device}")

    X_t = torch.tensor(X_train, dtype=torch.float32)
    y_t = torch.tensor(y_train, dtype=torch.float32)

    if X_val is None or y_val is None:
        # fallback internal split — prefer passing explicit val from
        # generate_drift_data.py so the split is image-level clean
        n_val = int(len(X_t) * 0.2)
        perm = torch.randperm(len(X_t))
        val_idx, train_idx = perm[:n_val], perm[n_val:]
        X_val_t, y_val_t = X_t[val_idx], y_t[val_idx]
        X_t, y_t = X_t[train_idx], y_t[train_idx]
    else:
        X_val_t = torch.tensor(X_val, dtype=torch.float32)
        y_val_t = torch.tensor(y_val, dtype=torch.float32)

    # pin the shuffle order to seed so training is fully reproducible
    _g = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        TensorDataset(X_t, y_t), batch_size=batch_size, shuffle=True, generator=_g
    )
    val_loader = DataLoader(
        TensorDataset(X_val_t, y_val_t), batch_size=batch_size
    )

    model = DriftMLP(input_dim=X_train.shape[1], dropout=dropout).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    pw = torch.tensor([pos_weight], dtype=torch.float32).to(device) if pos_weight else None
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pw)

    best_val_loss = float("inf")
    best_state = None

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_samples = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
            train_loss += loss.item() * len(xb)  # accumulate sum, not mean
            train_samples += len(xb)

        model.eval()
        val_loss = 0.0
        val_samples = 0
        with torch.inference_mode():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                val_loss += loss_fn(model(xb), yb).item() * len(xb)
                val_samples += len(xb)

        # divide by total samples for correct per-sample average loss
        avg_train = train_loss / train_samples
        avg_val = val_loss / val_samples

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if (epoch + 1) % 10 == 0:
            print(f"epoch {epoch+1}/{epochs}  train={avg_train:.4f}  val={avg_val:.4f}")

        if use_wandb:
            wandb.log({"epoch": epoch + 1, "train_loss": avg_train, "val_loss": avg_val})

    if use_wandb:
        wandb.finish()

    if best_state is None:
        # only reachable if epochs=0
        raise ValueError("train_model called with epochs=0 — no training occurred")

    model.load_state_dict(best_state)
    model.eval()
    return model


def save_model(model: DriftMLP, path: str):
    torch.save({
        "state_dict": model.state_dict(),
        "input_dim": model.input_dim,
        "dropout": model.dropout,
    }, path)


def load_model(path: str, device: torch.device | None = None) -> DriftMLP:
    if device is None:
        device = _get_device()
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    if "input_dim" not in checkpoint:
        raise ValueError(
            f"checkpoint at '{path}' is missing 'input_dim' — "
            "cannot safely infer architecture. Re-save with save_model()."
        )
    model = DriftMLP(
        input_dim=checkpoint["input_dim"],
        dropout=checkpoint.get("dropout", 0.2),
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()
    return model