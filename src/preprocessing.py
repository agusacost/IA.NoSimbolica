import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Elimina filas con nulos en columnas críticas."""
    return df.dropna(subset=["age", "embarked"]).reset_index(drop=True)


def encode_features(df: pd.DataFrame):
    """
    Codifica variables categóricas y retorna (X_array, y_array).
    sex: male=0, female=1
    embarked: one-hot (S, C, Q)
    Aplica StandardScaler sobre features numéricas.
    """
    data = df.copy()
    data["sex"] = data["sex"].map({"male": 0, "female": 1})
    embarked_dummies = pd.get_dummies(data["embarked"], prefix="embarked")
    data = pd.concat([data.drop(columns=["embarked"]), embarked_dummies], axis=1)

    y = data["survived"].values.astype(int)
    X_df = data.drop(columns=["survived"])

    scaler = StandardScaler()
    X = scaler.fit_transform(X_df.values.astype(float))
    return X, y


def split_data(X: np.ndarray, y: np.ndarray):
    """
    Divide en train(60%) / val(20%) / test(20%).
    Retorna: X_train, X_val, X_test, y_train, y_val, y_test
    """
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.4, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def apply_smote(X_train: np.ndarray, y_train: np.ndarray):
    """Aplica SMOTE para balancear clases en el conjunto de entrenamiento."""
    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X_train, y_train)
    return X_res, y_res
