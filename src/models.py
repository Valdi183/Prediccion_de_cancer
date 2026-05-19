"""
src/models.py

Entrenamiento de los 4 modelos de clasificación:
  1. Logistic Regression  — baseline lineal, class_weight='balanced'
  2. Random Forest        — 300 árboles, class_weight='balanced'
  3. XGBoost              — gradient boosting, scale_pos_weight=4.185
  4. MLP (PyTorch)        — 3 capas ocultas, Dropout, BCEWithLogitsLoss,
                            Early Stopping manual sobre val_loss (patience=12)

Nota: torch se importa de forma lazy (solo dentro de train_mlp) para que
los modelos sklearn puedan usarse aunque PyTorch no cargue en el entorno.
"""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier


def train_logistic_regression(X_train, y_train):
    model = LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        solver="lbfgs",
        C=1.0,
        random_state=42,
    )
    model.fit(X_train, y_train)
    return model


def train_random_forest(X_train, y_train):
    model = RandomForestClassifier(
        n_estimators=300,
        class_weight="balanced",
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model


def train_xgboost(X_train, y_train, scale_pos_weight=4.185):
    model = XGBClassifier(
        n_estimators=400,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model


def train_mlp(
    X_train, y_train, X_val, y_val,
    pos_weight,
    hidden_dims=(256, 128, 64),
    dropout=0.3,
    lr=1e-3,
    batch_size=2048,
    max_epochs=200,
    patience=25,
    print_every=10,
):
    """
    Entrena el MLP con Early Stopping manual sobre val_loss.
    Devuelve (model, history) donde history = {"train_loss": [...], "val_loss": [...]}.
    torch se importa aqui para no bloquear los modelos sklearn si PyTorch falla.
    """
    import torch
    import torch.nn as nn
    torch.set_num_threads(1)       # evita deadlock MKL/OpenMP en CPU Windows
    torch.set_num_interop_threads(1)

    class MLP(nn.Module):
        """3 capas ocultas con BatchNorm y Dropout. Salida: logit sin sigmoide."""

        def __init__(self, input_dim):
            super().__init__()
            layers = []
            prev = input_dim
            for h in hidden_dims:
                layers += [
                    nn.Linear(prev, h),
                    nn.BatchNorm1d(h),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ]
                prev = h
            layers.append(nn.Linear(prev, 1))
            self.net = nn.Sequential(*layers)

        def forward(self, x):
            return self.net(x).squeeze(1)

    device = torch.device("cpu")

    X_tr = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_tr = torch.tensor(y_train, dtype=torch.float32).to(device)
    X_v  = torch.tensor(X_val,   dtype=torch.float32).to(device)
    y_v  = torch.tensor(y_val,   dtype=torch.float32).to(device)

    model     = MLP(X_train.shape[1]).to(device)
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([pos_weight], dtype=torch.float32).to(device)
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.7)

    dataset = torch.utils.data.TensorDataset(X_tr, y_tr)
    loader  = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    history = {"train_loss": [], "val_loss": []}
    best_val_loss = float("inf")
    best_weights  = None
    wait          = 0

    for epoch in range(max_epochs):
        model.train()
        batch_losses = []
        for Xb, yb in loader:
            optimizer.zero_grad()
            loss = criterion(model(Xb), yb)
            loss.backward()
            optimizer.step()
            batch_losses.append(loss.item())

        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(X_v), y_v).item()

        epoch_train = float(np.mean(batch_losses))
        history["train_loss"].append(epoch_train)
        history["val_loss"].append(val_loss)
        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_weights  = {k: v.clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print(f"  Early stopping en epoca {epoch + 1}  (mejor val_loss = {best_val_loss:.4f})", flush=True)
                break

        if print_every and (epoch + 1) % print_every == 0:
            print(f"  ep {epoch+1:3d}/{max_epochs}  train={epoch_train:.4f}  val={val_loss:.4f}  patience={wait}/{patience}", flush=True)

    model.load_state_dict(best_weights)
    return model, history
