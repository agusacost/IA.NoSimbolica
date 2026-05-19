import os
import pandas as pd

from src.data_loader import load_titanic, describe_dataset
from src.preprocessing import clean_data, encode_features, split_data, apply_smote
from src.models import build_model_1, build_model_2, build_model_3, MODEL_CONFIGS
from src.evaluation import train_and_evaluate, build_summary_table
from src.visualization import (
    plot_loss_curves,
    plot_confusion_matrices,
    plot_metrics_comparison,
    plot_class_balance,
)

os.makedirs("outputs", exist_ok=True)


def main():
    # --- 1. Carga y exploración ---
    print("=" * 60)
    print("1. CARGA DE DATOS")
    print("=" * 60)
    df = load_titanic()
    info = describe_dataset(df)
    print(f"  Shape: {info['shape']}")
    print(f"  Distribución de clases: {info['class_distribution']}")
    print(f"  Valores nulos:\n{pd.Series(info['missing_values'])}")

    # --- 2. Preprocesamiento ---
    print("\n" + "=" * 60)
    print("2. PREPROCESAMIENTO")
    print("=" * 60)
    df_clean = clean_data(df)
    print(f"  Filas tras limpieza: {len(df_clean)} (original: {len(df)})")

    X, y = encode_features(df_clean)
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)
    print(f"  Train: {len(y_train)} | Val: {len(y_val)} | Test: {len(y_test)}")

    import numpy as np
    before_counts = np.bincount(y_train)
    print(f"  Distribución train antes SMOTE: No sobrevivió={before_counts[0]}, Sobrevivió={before_counts[1]}")
    X_train_bal, y_train_bal = apply_smote(X_train, y_train)
    after_counts = np.bincount(y_train_bal)
    print(f"  Distribución train después SMOTE: No sobrevivió={after_counts[0]}, Sobrevivió={after_counts[1]}")

    plot_class_balance(y_train, y_train_bal)

    # --- 3. Entrenamiento y evaluación ---
    print("\n" + "=" * 60)
    print("3. ENTRENAMIENTO DE MODELOS")
    print("=" * 60)

    builders = [build_model_1, build_model_2, build_model_3]
    results = []

    for i, (builder, config) in enumerate(zip(builders, MODEL_CONFIGS)):
        print(f"\n  [{i+1}] {config['name']}")
        print(f"      {config['description']}")
        model = builder()
        result = train_and_evaluate(model, X_train_bal, y_train_bal, X_val, y_val)
        result["name"] = config["name"]
        results.append(result)

        tm = result["train_metrics"]
        vm = result["val_metrics"]
        print(f"      Train  → Acc={tm['accuracy']:.4f}  F1={tm['f1_score']:.4f}  Tiempo={tm['train_time_s']:.2f}s")
        print(f"      Val    → Acc={vm['accuracy']:.4f}  F1={vm['f1_score']:.4f}  Prec={vm['precision']:.4f}  Rec={vm['recall']:.4f}")

    # --- 4. Evaluación en Test del mejor modelo ---
    print("\n" + "=" * 60)
    print("4. EVALUACIÓN EN TEST (mejor modelo por F1 en validación)")
    print("=" * 60)

    best_result = max(results, key=lambda r: r["val_metrics"]["f1_score"])
    best_model = best_result["model"]
    from src.evaluation import compute_metrics
    import time
    t0 = time.time()
    y_pred_test = best_model.predict(X_test)
    test_metrics = compute_metrics(y_test, y_pred_test, time.time() - t0)
    print(f"  Mejor modelo: {best_result['name']}")
    print(f"  Test → Acc={test_metrics['accuracy']:.4f}  F1={test_metrics['f1_score']:.4f}  Prec={test_metrics['precision']:.4f}  Rec={test_metrics['recall']:.4f}")

    # --- 5. Tabla resumen ---
    print("\n" + "=" * 60)
    print("5. TABLA RESUMEN (Validación)")
    print("=" * 60)
    summary = build_summary_table(results)
    print(summary.to_string(index=False))
    summary.to_csv("outputs/results_summary.csv", index=False)
    print("\n  Tabla guardada en outputs/results_summary.csv")

    # --- 6. Gráficas ---
    print("\n" + "=" * 60)
    print("6. GENERANDO GRÁFICAS")
    print("=" * 60)
    trained_models = [r["model"] for r in results]
    names = [r["name"] for r in results]
    plot_loss_curves(trained_models, names)
    plot_confusion_matrices(results)
    plot_metrics_comparison(summary)
    print("  Gráficas guardadas en outputs/")

    # --- 7. Conclusión ---
    print("\n" + "=" * 60)
    print("7. CONCLUSIÓN")
    print("=" * 60)
    print(f"  El modelo con mejor desempeño en validación es: {best_result['name']}")
    vm = best_result["val_metrics"]
    print(f"  F1={vm['f1_score']:.4f}  Accuracy={vm['accuracy']:.4f}  Precision={vm['precision']:.4f}  Recall={vm['recall']:.4f}")


if __name__ == "__main__":
    main()
