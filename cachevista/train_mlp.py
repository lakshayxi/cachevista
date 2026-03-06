import numpy as np

from cachevista.mlp import train_model, save_model

if __name__ == "__main__":
    X = np.load("data/drift_X.npy")
    y = np.load("data/drift_y.npy")

    print(f"training on {len(X)} pairs, input dim: {X.shape[1]}")
    model = train_model(X, y, epochs=30)
    save_model(model, "models/drift_mlp.pt")
    print("saved to models/drift_mlp.pt")
