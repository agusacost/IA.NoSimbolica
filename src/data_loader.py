import pandas as pd
import seaborn as sns


def load_titanic() -> pd.DataFrame:
    """Carga el dataset Titanic desde seaborn y retorna columnas relevantes."""
    df = sns.load_dataset("titanic")
    cols = ["survived", "pclass", "sex", "age", "sibsp", "parch", "fare", "embarked"]
    return df[cols].copy()


def describe_dataset(df: pd.DataFrame) -> dict:
    """Retorna estadísticas descriptivas básicas del dataset."""
    return {
        "shape": df.shape,
        "class_distribution": df["survived"].value_counts().to_dict(),
        "missing_values": df.isnull().sum().to_dict(),
    }
