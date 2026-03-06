import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class DriftMLP(nn.Module):
    def __init__(self, input_dim=1536):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x).squeeze(1)

    def predict(self, emb_prev: np.ndarray, emb_curr: np.ndarray) -> float:
        x = torch.tensor(np.concatenate([emb_prev, emb_curr]), dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            return float(self.forward(x))


def train_model(X: np.ndarray, y: np.ndarray, epochs=30, lr=1e-3) -> DriftMLP:
    loader = DataLoader(
        TensorDataset(torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)),
        batch_size=8,
        shuffle=True,
    )
    model = DriftMLP()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCELoss()

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for xb, yb in loader:
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
            epoch_loss += loss.item()
        if (epoch + 1) % 10 == 0:
            print(f"epoch {epoch+1}/{epochs}  loss={epoch_loss/len(loader):.4f}")

    model.eval()
    return model


def save_model(model: DriftMLP, path: str):
    torch.save(model.state_dict(), path)


def load_model(path: str) -> DriftMLP:
    model = DriftMLP()
    model.load_state_dict(torch.load(path, weights_only=True))
    model.eval()
    return model
