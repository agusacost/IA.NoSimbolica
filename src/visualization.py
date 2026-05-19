import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neural_network import MLPClassifier

OUTPUT_DIR = "outputs"


def plot_loss_curves(models: list, names: list, save: bool = True):
    """Grafica la curva de pérdida de entrenamiento para cada modelo."""
    fig, axes = plt.subplots(1, len(models), figsize=(5 * len(models), 4))
    if len(models) == 1:
        axes = [axes]
    for ax, model, name in zip(axes, models, names):
        if hasattr(model, "loss_curve_"):
            ax.plot(model.loss_curve_, label="Train Loss", color="steelblue")
            ax.set_title(name, fontsize=10)
            ax.set_xlabel("Época")
            ax.set_ylabel("Pérdida")
            ax.legend()
    plt.suptitle("Curvas de Pérdida por Modelo", fontsize=13, fontweight="bold")
    plt.tight_layout()
    if save:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        plt.savefig(f"{OUTPUT_DIR}/loss_curves.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_confusion_matrices(results: list, save: bool = True):
    """Grafica matrices de confusión para validación de cada modelo."""
    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4))
    if n == 1:
        axes = [axes]
    for ax, r in zip(axes, results):
        cm = np.array(r["val_metrics"]["confusion_matrix"])
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues", ax=ax,
            xticklabels=["No Sobrevivió", "Sobrevivió"],
            yticklabels=["No Sobrevivió", "Sobrevivió"],
        )
        ax.set_title(r["name"], fontsize=9)
        ax.set_xlabel("Predicción")
        ax.set_ylabel("Real")
    plt.suptitle("Matrices de Confusión (Validación)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    if save:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        plt.savefig(f"{OUTPUT_DIR}/confusion_matrices.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_metrics_comparison(summary_df, save: bool = True):
    """Gráfico de barras agrupadas comparando accuracy, precision, recall, f1."""
    metrics = ["accuracy", "precision", "recall", "f1_score"]
    x = np.arange(len(summary_df))
    width = 0.2
    fig, ax = plt.subplots(figsize=(10, 5))
    for i, metric in enumerate(metrics):
        ax.bar(x + i * width, summary_df[metric], width, label=metric.replace("_", " ").title())
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(summary_df["modelo"], rotation=10, fontsize=9)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Valor")
    ax.set_title("Comparación de Métricas por Modelo (Validación)", fontsize=13, fontweight="bold")
    ax.legend()
    plt.tight_layout()
    if save:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        plt.savefig(f"{OUTPUT_DIR}/metrics_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_class_balance(y_before, y_after, save: bool = True):
    """Muestra distribución de clases antes y después de SMOTE."""
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    for ax, y, title in zip(axes, [y_before, y_after], ["Antes de SMOTE", "Después de SMOTE"]):
        counts = np.bincount(y)
        ax.bar(["No Sobrevivió (0)", "Sobrevivió (1)"], counts, color=["salmon", "steelblue"])
        ax.set_title(title, fontsize=11)
        ax.set_ylabel("Cantidad de instancias")
        for j, v in enumerate(counts):
            ax.text(j, v + 1, str(v), ha="center", fontsize=10)
    plt.suptitle("Efecto del Balanceo con SMOTE", fontsize=13, fontweight="bold")
    plt.tight_layout()
    if save:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        plt.savefig(f"{OUTPUT_DIR}/class_balance.png", dpi=150, bbox_inches="tight")
    plt.close()
