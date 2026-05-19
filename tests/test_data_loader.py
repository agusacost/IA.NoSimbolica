import pandas as pd
from src.data_loader import load_titanic, describe_dataset


def test_load_titanic_returns_dataframe():
    df = load_titanic()
    assert isinstance(df, pd.DataFrame)


def test_load_titanic_has_required_columns():
    df = load_titanic()
    required = {"survived", "pclass", "sex", "age", "sibsp", "parch", "fare", "embarked"}
    assert required.issubset(set(df.columns))


def test_load_titanic_has_rows():
    df = load_titanic()
    assert len(df) > 800


def test_describe_dataset_returns_dict():
    df = load_titanic()
    info = describe_dataset(df)
    assert "shape" in info
    assert "class_distribution" in info
    assert "missing_values" in info
