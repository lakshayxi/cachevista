import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


def _get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _build_features(emb1: np.ndarray, emb2: np.ndarray) -> np.ndarray:
    """
    Explicit similarity features from two joint embeddings.
    Works for any embedding dimension — image-only (768) or joint (1152).
      |emb1 - emb2|  : per-dim difference magnitude
      emb1 * emb2    : element-wise product
      cosine_sim     : scalar dot product of unit-norm vectors
    Output dim = 2*input_dim + 1
    """
    diff = np.abs(emb1 - emb2)
    prod = emb1 * emb2
    cosine = np.array([np.dot(emb1, emb2)], dtype=np.float32)
    return np.concatenate([diff, prod, cosine])


class DriftMLP(nn.Module):
    def __init__(self, input_dim: int = 2305):
        super().__init__()
        self.input_dim = input_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(1)

    def predict(self, emb_query: np.ndarray, emb_candidate: np.ndarray) -> float:
        """
        Returns drift probability for (query_joint_emb, candidate_joint_emb).
        High score = drift = don't reuse cached response.
        Both embeddings must be unit-norm joint vectors.
        """
        assert emb_query.shape == emb_candidate.shape, (
            f"shape mismatch: {emb_query.shape} vs {emb_candidate.shape}"
        )
        return self.predict_features(_build_features(emb_query, emb_candidate))

    def predict_features(self, features: np.ndarray) -> float:
        x = torch.from_numpy(features).unsqueeze(0)
        with torch.no_grad():
            logit = self.forward(x)
        return float(torch.sigmoid(logit))


def train_model(X_train, y_train, X_val=None, y_val=None,
                epochs=50, lr=1e-3, batch_size=64, use_wandb=False):
    if use_wandb:
        import wandb
        wandb.init(project="cachevista", config={"epochs": epochs, "lr": lr, "batch_size": batch_size})

    device = _get_device()
    print(f"training on {device}")

    X_t = torch.tensor(X_train, dtype=torch.float32)
    y_t = torch.tensor(y_train, dtype=torch.float32)

    if X_val is None or y_val is None:
        n_val = int(len(X_t) * 0.2)
        perm = torch.randperm(len(X_t))
        val_idx, train_idx = perm[:n_val], perm[n_val:]
        X_val_t, y_val_t = X_t[val_idx], y_t[val_idx]
        X_t, y_t = X_t[train_idx], y_t[train_idx]
    else:
        X_val_t = torch.tensor(X_val, dtype=torch.float32)
        y_val_t = torch.tensor(y_val, dtype=torch.float32)

    train_loader = DataLoader(TensorDataset(X_t, y_t), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val_t, y_val_t), batch_size=batch_size)

    input_dim = X_train.shape[1]
    model = DriftMLP(input_dim=input_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()

    best_val_loss = float("inf")
    best_state = None

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
            train_loss += loss.item()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                val_loss += loss_fn(model(xb), yb).item()

        avg_train = train_loss / len(train_loader)
        avg_val = val_loss / len(val_loader)

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if (epoch + 1) % 10 == 0:
            print(f"epoch {epoch+1}/{epochs}  train={avg_train:.4f}  val={avg_val:.4f}")

        if use_wandb:
            import wandb
            wandb.log({"epoch": epoch + 1, "train_loss": avg_train, "val_loss": avg_val})

    if use_wandb:
        import wandb
        wandb.finish()

    model.load_state_dict(best_state)
    model.eval()
    return model


def save_model(model: DriftMLP, path: str):
    torch.save({"state_dict": model.state_dict(), "input_dim": model.input_dim}, path)


def load_model(path: str) -> DriftMLP:
    checkpoint = torch.load(path, weights_only=True)
    if "input_dim" in checkpoint:
        model = DriftMLP(input_dim=checkpoint["input_dim"])
        model.load_state_dict(checkpoint["state_dict"])
    else:
        model = DriftMLP()
        model.load_state_dict(checkpoint)
    model.eval()
    return model