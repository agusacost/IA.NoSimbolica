import numpy as np
import time
from src.evaluation import compute_metrics, train_and_evaluate, build_summary_table
from src.models import build_model_1


def test_compute_metrics_returns_expected_keys():
    y_true = np.array([0, 1, 0, 1, 1, 0])
    y_pred = np.array([0, 1, 0, 0, 1, 1])
    metrics = compute_metrics(y_true, y_pred, train_time=1.23)
    assert "accuracy" in metrics
    assert "precision" in metrics
    assert "recall" in metrics
    assert "f1_score" in metrics
    assert "train_time_s" in metrics
    assert "confusion_matrix" in metrics


def test_compute_metrics_accuracy_range():
    y_true = np.array([0, 1, 0, 1])
    y_pred = np.array([0, 1, 0, 1])
    metrics = compute_metrics(y_true, y_pred, train_time=0.5)
    assert metrics["accuracy"] == 1.0


def test_train_and_evaluate_returns_metrics_and_model():
    X = np.random.rand(200, 8)
    y = np.random.randint(0, 2, 200)
    X_val = np.random.rand(50, 8)
    y_val = np.random.randint(0, 2, 50)
    model = build_model_1()
    result = train_and_evaluate(model, X, y, X_val, y_val)
    assert "train_metrics" in result
    assert "val_metrics" in result
    assert "model" in result


def test_build_summary_table_has_all_models():
    results = [
        {"name": "M1", "val_metrics": {"accuracy": 0.8, "precision": 0.79, "recall": 0.81, "f1_score": 0.80, "train_time_s": 1.0}},
        {"name": "M2", "val_metrics": {"accuracy": 0.85, "precision": 0.84, "recall": 0.86, "f1_score": 0.85, "train_time_s": 2.0}},
        {"name": "M3", "val_metrics": {"accuracy": 0.82, "precision": 0.81, "recall": 0.83, "f1_score": 0.82, "train_time_s": 1.5}},
    ]
    table = build_summary_table(results)
    assert len(table) == 3
    assert "accuracy" in table.columns
