# TP 5.1 — Red Neuronal Artificial para Clasificación en Titanic

**Materia:** Inteligencia Artificial No Simbólica  
**Tema:** IA Conexionista — Redes Neuronales Artificiales (RNA)  
**Dataset:** Titanic (cargado desde seaborn)  
**Tarea:** Clasificación binaria — predecir si un pasajero sobrevivió (`survived = 1`) o no (`survived = 0`)

---

## 1. Tipo de Red Neuronal Utilizada

Se usa un **Perceptrón Multicapa (MLP — Multi-Layer Perceptron)**, implementado con `MLPClassifier` de scikit-learn.

El MLP es una red neuronal de propagación hacia adelante (feedforward) totalmente conectada, compuesta por:

- **Capa de entrada:** recibe los 9 atributos del dataset procesado.
- **Capas ocultas:** una o más capas con neuronas no lineales (ReLU o tanh).
- **Capa de salida:** una neurona con activación logística que produce probabilidades para cada clase.

El aprendizaje es **supervisado**: los pesos de la red se ajustan mediante backpropagation, minimizando la pérdida de entropía cruzada usando descenso por gradiente (SGD) o Adam.

---

## 2. Pipeline General

```
Dataset Titanic (891 filas)
        |
        v
  [ 1. Carga ]
  load_titanic()       → selecciona 8 columnas relevantes
        |
        v
  [ 2. Limpieza ]
  clean_data()         → elimina filas con nulos en 'age' y 'embarked'
        |
        v
  [ 3. Encoding + Escalado ]
  encode_features()    → codifica categorías, aplica StandardScaler
        |
        v
  [ 4. Partición ]
  split_data()         → Train 60% / Val 20% / Test 20%
        |
        v
  [ 5. Balanceo SMOTE ]
  apply_smote()        → solo sobre el conjunto de entrenamiento
        |
        v
  [ 6. Entrenamiento ]
  train_and_evaluate() → entrena 3 modelos MLP distintos
        |
        v
  [ 7. Evaluación ]
  compute_metrics()    → Accuracy, Precision, Recall, F1, Matriz de Confusión
        |
        v
  [ 8. Selección del mejor modelo ]
  → criterio: mayor F1-Score en validación
        |
        v
  [ 9. Evaluación en Test ]
  → métricas finales sobre datos no vistos
        |
        v
  [ 10. Visualización ]
  → gráficas guardadas en outputs/
```

---

## 3. Preprocesamiento

### 3.1 Carga del Dataset (`src/data_loader.py`)

```python
def load_titanic() -> pd.DataFrame:
    df = sns.load_dataset("titanic")
    cols = ["survived", "pclass", "sex", "age", "sibsp", "parch", "fare", "embarked"]
    return df[cols].copy()
```

Se seleccionan 8 columnas de las 15 disponibles en el dataset de seaborn. Las elegidas son las que representan atributos directamente relacionados con la supervivencia según literatura del dominio.

| Columna   | Tipo       | Descripción                                 |
|-----------|------------|---------------------------------------------|
| survived  | int (0/1)  | Variable objetivo (clase)                   |
| pclass    | int (1-3)  | Clase del pasaje (1ra, 2da, 3ra)            |
| sex       | string     | Sexo del pasajero                           |
| age       | float      | Edad en años                                |
| sibsp     | int        | Hermanos/cónyuge a bordo                    |
| parch     | int        | Padres/hijos a bordo                        |
| fare      | float      | Tarifa pagada                               |
| embarked  | string     | Puerto de embarque (S, C, Q)                |

### 3.2 Limpieza (`src/preprocessing.py` — `clean_data`)

```python
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    return df.dropna(subset=["age", "embarked"]).reset_index(drop=True)
```

Se eliminan las filas con valores nulos únicamente en las columnas `age` y `embarked`, que son las que presentan datos faltantes en el dataset original. Esto reduce el dataset de 891 a aproximadamente 712 filas.

### 3.3 Codificación y Escalado (`encode_features`)

```python
def encode_features(df: pd.DataFrame):
    data = df.copy()
    # Codificación binaria de sexo
    data["sex"] = data["sex"].map({"male": 0, "female": 1})
    # One-hot encoding del puerto de embarque
    embarked_dummies = pd.get_dummies(data["embarked"], prefix="embarked")
    data = pd.concat([data.drop(columns=["embarked"]), embarked_dummies], axis=1)

    y = data["survived"].values.astype(int)
    X_df = data.drop(columns=["survived"])

    scaler = StandardScaler()
    X = scaler.fit_transform(X_df.values.astype(float))
    return X, y
```

Tres transformaciones clave:

1. **`sex`** → codificación binaria: `male=0`, `female=1`.
2. **`embarked`** → one-hot encoding: genera columnas `embarked_S`, `embarked_C`, `embarked_Q`. Esto evita introducir un orden ordinal falso entre los puertos.
3. **StandardScaler** → normaliza cada feature a media 0 y desvío estándar 1. Esto es esencial para el MLP ya que el gradiente converge más rápido y de forma más estable cuando las features están en la misma escala.

El vector de entrada resultante tiene **9 dimensiones** (pclass, sex, age, sibsp, parch, fare, embarked_C, embarked_Q, embarked_S).

### 3.4 Partición del Dataset (`split_data`)

```python
def split_data(X, y):
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.4, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )
    return X_train, X_val, X_test, y_train, y_val, y_test
```

División en tres conjuntos con `stratify=y` para mantener la proporción de clases original en cada partición:

| Conjunto    | Proporción | Uso                                           |
|-------------|------------|-----------------------------------------------|
| Train       | 60%        | Ajuste de pesos (entrenamiento)               |
| Validación  | 20%        | Selección del mejor modelo                    |
| Test        | 20%        | Evaluación final del modelo ganador           |

El parámetro `stratify=y` garantiza que la distribución de sobrevivientes/no sobrevivientes sea proporcional en los tres splits.

### 3.5 Balanceo con SMOTE (`apply_smote`)

```python
def apply_smote(X_train, y_train):
    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X_train, y_train)
    return X_res, y_res
```

El dataset Titanic tiene desbalance de clases (más no-sobrevivientes que sobrevivientes). **SMOTE (Synthetic Minority Over-sampling Technique)** genera muestras sintéticas de la clase minoritaria interpolando entre instancias reales existentes, hasta igualar ambas clases.

**Solo se aplica sobre el conjunto de entrenamiento**, nunca sobre validación o test. Aplicarlo sobre los otros conjuntos contaminaría la evaluación al incluir datos artificiales mezclados con reales.

---

## 4. Los Tres Modelos MLP

Todos los modelos usan `MLPClassifier` de scikit-learn y varían en arquitectura, función de activación, optimizador, tasa de aprendizaje y épocas.

### Modelo 1 — Baseline MLP (`src/models.py`)

```python
MLPClassifier(
    hidden_layer_sizes=(64,),
    activation="relu",
    solver="sgd",
    learning_rate_init=0.01,
    max_iter=100,
    random_state=42,
)
```

| Parámetro       | Valor  | Descripción                                  |
|-----------------|--------|----------------------------------------------|
| Arquitectura    | 9→64→1 | 1 capa oculta con 64 neuronas                |
| Activación      | ReLU   | `f(x) = max(0, x)`, evita el problema del gradiente evanescente |
| Optimizador     | SGD    | Descenso por gradiente estocástico, simple y rápido |
| Tasa aprendizaje| 0.01   | Moderada para SGD                            |
| Épocas          | 100    | Número máximo de iteraciones                 |

Es el modelo más sencillo, sirve como referencia base (baseline).

### Modelo 2 — MLP Profundo

```python
MLPClassifier(
    hidden_layer_sizes=(128, 64, 32),
    activation="relu",
    solver="adam",
    learning_rate_init=0.001,
    max_iter=200,
    random_state=42,
)
```

| Parámetro       | Valor           | Descripción                                       |
|-----------------|-----------------|---------------------------------------------------|
| Arquitectura    | 9→128→64→32→1   | 3 capas ocultas, mayor capacidad representacional |
| Activación      | ReLU            | Misma que M1                                      |
| Optimizador     | Adam            | Adapta la tasa de aprendizaje por parámetro, converge más rápido |
| Tasa aprendizaje| 0.001           | Más baja por usar Adam                            |
| Épocas          | 200             | Más iteraciones para la red más compleja          |

Red más profunda con mayor capacidad para capturar patrones no lineales complejos.

### Modelo 3 — MLP Regularizado

```python
MLPClassifier(
    hidden_layer_sizes=(64, 32),
    activation="tanh",
    solver="adam",
    learning_rate_init=0.001,
    max_iter=200,
    alpha=0.01,
    random_state=42,
)
```

| Parámetro       | Valor    | Descripción                                         |
|-----------------|----------|-----------------------------------------------------|
| Arquitectura    | 9→64→32→1| 2 capas ocultas, tamaño intermedio                  |
| Activación      | tanh     | Salida en [-1, 1], centrada en cero                 |
| Optimizador     | Adam     | Igual que M2                                        |
| Tasa aprendizaje| 0.001    | Igual que M2                                        |
| Épocas          | 200      | Igual que M2                                        |
| alpha (L2)      | 0.01     | Regularización L2 para penalizar pesos grandes y reducir sobreajuste |

Agrega **regularización L2** (`alpha`), que actúa como penalización sobre los pesos durante el entrenamiento: `Loss_total = Loss_CE + alpha * sum(w²)`. Esto reduce el sobreajuste en datasets pequeños.

---

## 5. Entrenamiento y Evaluación (`src/evaluation.py`)

```python
def train_and_evaluate(model, X_train, y_train, X_val, y_val):
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
```

Cada modelo se entrena sobre el conjunto de entrenamiento balanceado con SMOTE y se evalúa tanto en entrenamiento como en validación para detectar posible sobreajuste.

### Métricas calculadas

```python
def compute_metrics(y_true, y_pred, train_time):
    return {
        "accuracy":         accuracy_score(y_true, y_pred),
        "precision":        precision_score(y_true, y_pred),
        "recall":           recall_score(y_true, y_pred),
        "f1_score":         f1_score(y_true, y_pred),
        "train_time_s":     train_time,
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }
```

| Métrica    | Fórmula                          | Qué mide                                           |
|------------|----------------------------------|----------------------------------------------------|
| Accuracy   | (TP+TN) / total                  | Proporción global de predicciones correctas        |
| Precision  | TP / (TP+FP)                     | De los predichos positivos, cuántos son reales     |
| Recall     | TP / (TP+FN)                     | De los positivos reales, cuántos fueron detectados |
| F1-Score   | 2·(Prec·Rec)/(Prec+Rec)          | Media armónica entre Precision y Recall            |

El **F1-Score** es la métrica principal de selección porque balancea Precision y Recall, siendo más robusta que la Accuracy cuando hay desbalance de clases.

---

## 6. Selección del Mejor Modelo

```python
best_result = max(results, key=lambda r: r["val_metrics"]["f1_score"])
```

El modelo con mayor F1-Score en validación es seleccionado como ganador y se evalúa en el conjunto de test (datos nunca vistos durante el entrenamiento ni la selección).

---

## 7. Visualizaciones (`src/visualization.py`)

| Función                  | Archivo generado             | Contenido                                         |
|--------------------------|------------------------------|---------------------------------------------------|
| `plot_class_balance()`   | `class_balance.png`          | Barras antes/después de SMOTE                     |
| `plot_loss_curves()`     | `loss_curves.png`            | Curva de pérdida por época para cada modelo       |
| `plot_confusion_matrices()` | `confusion_matrices.png`  | Matrices de confusión de validación               |
| `plot_metrics_comparison()` | `metrics_comparison.png`  | Barras agrupadas: Acc, Prec, Rec, F1 por modelo   |

Todos los gráficos usan `matplotlib.use('Agg')` para generarse sin interfaz gráfica y se guardan en `outputs/`.

---

## 8. Estructura de Archivos

```
TP5.1/
├── main.py                  # Orquesta todo el pipeline
├── requirements.txt         # Dependencias
├── src/
│   ├── data_loader.py       # Carga del dataset Titanic
│   ├── preprocessing.py     # Limpieza, encoding, SMOTE, split
│   ├── models.py            # Definición de los 3 modelos MLP
│   ├── evaluation.py        # Entrenamiento, métricas, tabla resumen
│   └── visualization.py     # Generación de gráficas
├── tests/
│   ├── test_data_loader.py
│   ├── test_preprocessing.py
│   ├── test_models.py
│   └── test_evaluation.py
└── outputs/
    ├── class_balance.png
    ├── loss_curves.png
    ├── confusion_matrices.png
    ├── metrics_comparison.png
    └── results_summary.csv
```

---

## 9. Dependencias

| Librería          | Versión mínima | Rol en el proyecto                              |
|-------------------|----------------|-------------------------------------------------|
| scikit-learn      | 1.4.0          | MLPClassifier, métricas, StandardScaler, split  |
| imbalanced-learn  | 0.12.0         | SMOTE para balanceo de clases                   |
| pandas            | 2.2.0          | Manipulación de datos tabulares                 |
| numpy             | 1.26.0         | Operaciones numéricas sobre arrays              |
| matplotlib        | 3.8.0          | Generación de gráficas                          |
| seaborn           | 0.13.0         | Carga del dataset Titanic, heatmaps             |

---

## 10. Cómo Ejecutar

```bash
# Instalar dependencias
pip install -r requirements.txt

# Ejecutar el pipeline completo
python main.py

# Ejecutar los tests
pytest tests/ -v
```

La ejecución de `main.py` imprime 7 secciones numeradas en consola y genera 4 imágenes y 1 CSV en la carpeta `outputs/`.
