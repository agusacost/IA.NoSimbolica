import time
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.neural_network import MLPClassifier


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, train_time: float) -> dict:
    """Calcula accuracy, precision, recall, f1, tiempo y matriz de confusión."""
    return {
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "recall": round(recall_score(y_true, y_pred, zero_division=0), 4),
        "f1_score": round(f1_score(y_true, y_pred, zero_division=0), 4),
        "train_time_s": round(train_time, 4),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }


def train_and_evaluate(
    model: MLPClassifier,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
) -> dict:
    """Entrena el modelo y retorna métricas de train y validación."""
    t0 = time.time()
    model.fit(X_train, y_train)
    elapsed = time.time() - t0

    y_pred_train = model.predict(X_train)
    y_pred_val = model.predict(X_val)

    return {
        "model": model,
        "train_metrics": compute_metrics(y_train, y_pred_train, elapsed),
        "val_metrics": compute_metrics(y_val, y_pred_val, elapsed),
    }


def build_summary_table(results: list) -> pd.DataFrame:
    """
    Recibe lista de dicts con 'name' y 'val_metrics'.
    Retorna DataFrame comparativo con métricas de validación.
    """
    rows = []
    for r in results:
        row = {"modelo": r["name"]}
        row.update({k: v for k, v in r["val_metrics"].items() if k != "confusion_matrix"})
        rows.append(row)
    return pd.DataFrame(rows)
