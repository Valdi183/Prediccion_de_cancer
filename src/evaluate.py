"""
src/evaluate.py

Funciones de evaluacion compartidas:
  - optimize_threshold : barre umbrales [0.05, 0.95] maximizando F1 en validacion
  - eval_model         : Accuracy, Precision, Recall, F1, AUC-ROC con umbral fijo
  - plot_roc_pr        : curvas ROC y Precision-Recall comparativas
  - plot_confusion     : matrices de confusion en cuadricula

torch se importa de forma lazy en _get_proba para no bloquear la evaluacion
de modelos sklearn cuando PyTorch no esta disponible en el entorno.
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score,
    roc_curve, precision_recall_curve, confusion_matrix,
)


def _get_proba(model, X):
    """Probabilidad de clase positiva para sklearn o nn.Module de PyTorch."""
    # Intento lazy de PyTorch — falla sin error si no esta disponible
    try:
        import torch
        import torch.nn as nn
        if isinstance(model, nn.Module):
            model.eval()
            with torch.no_grad():
                logits = model(torch.tensor(X, dtype=torch.float32))
                return torch.sigmoid(logits).numpy()
    except (ImportError, OSError):
        pass
    return model.predict_proba(X)[:, 1]


def optimize_threshold(model, X_val, y_val, thresholds=None):
    """
    Devuelve (best_threshold, best_f1) maximizando F1 en validacion.
    Umbral fijado siempre sobre val — nunca sobre test.
    """
    if thresholds is None:
        thresholds = np.arange(0.05, 0.96, 0.005)

    proba = _get_proba(model, X_val)
    best_t, best_f1 = 0.5, 0.0
    for t in thresholds:
        preds = (proba >= t).astype(int)
        f = f1_score(y_val, preds, zero_division=0)
        if f > best_f1:
            best_f1 = f
            best_t  = t
    return float(best_t), float(best_f1)


def eval_model(model, X, y, threshold):
    """
    Evalua el modelo con el umbral fijado.
    Devuelve dict con Accuracy, Precision, Recall, F1, AUC-ROC.
    """
    proba = _get_proba(model, X)
    preds = (proba >= threshold).astype(int)
    return {
        "Accuracy":  accuracy_score(y, preds),
        "Precision": precision_score(y, preds, zero_division=0),
        "Recall":    recall_score(y, preds, zero_division=0),
        "F1":        f1_score(y, preds, zero_division=0),
        "AUC-ROC":   roc_auc_score(y, proba),
    }


def plot_roc_pr(models_dict, X, y, save_path=None):
    """Curvas ROC y Precision-Recall para todos los modelos."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    colors = plt.cm.tab10.colors

    for i, (name, model) in enumerate(models_dict.items()):
        proba = _get_proba(model, X)
        c = colors[i % len(colors)]

        fpr, tpr, _ = roc_curve(y, proba)
        auc = roc_auc_score(y, proba)
        axes[0].plot(fpr, tpr, color=c, lw=2, label=f"{name}  AUC={auc:.3f}")

        prec, rec, _ = precision_recall_curve(y, proba)
        axes[1].plot(rec, prec, color=c, lw=2, label=name)

    axes[0].plot([0, 1], [0, 1], "k--", lw=1)
    axes[0].set_xlabel("False Positive Rate")
    axes[0].set_ylabel("True Positive Rate")
    axes[0].set_title("Curva ROC")
    axes[0].legend()

    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].set_title("Curva Precision-Recall")
    axes[1].legend()

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_confusion(models_dict, X, y, thresholds_dict, save_path=None):
    """Matrices de confusion en cuadricula (una por modelo)."""
    n = len(models_dict)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4))
    if n == 1:
        axes = [axes]

    for ax, (name, model) in zip(axes, models_dict.items()):
        proba = _get_proba(model, X)
        preds = (proba >= thresholds_dict[name]).astype(int)
        cm = confusion_matrix(y, preds)
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues", ax=ax,
            xticklabels=["No cancer", "Cancer"],
            yticklabels=["No cancer", "Cancer"],
        )
        ax.set_title(f"{name}  (umbral={thresholds_dict[name]:.2f})")
        ax.set_xlabel("Prediccion")
        ax.set_ylabel("Real")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
