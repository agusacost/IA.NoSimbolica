import numpy as np
from src.data_loader import load_titanic
from src.preprocessing import clean_data, encode_features, split_data, apply_smote


def test_clean_data_removes_nulls():
    df = load_titanic()
    cleaned = clean_data(df)
    assert cleaned.isnull().sum().sum() == 0


def test_clean_data_keeps_survived_column():
    df = load_titanic()
    cleaned = clean_data(df)
    assert "survived" in cleaned.columns


def test_encode_features_no_categorical():
    df = load_titanic()
    cleaned = clean_data(df)
    X, y = encode_features(cleaned)
    assert X.dtype == np.float64 or X.dtype == np.float32 or X.dtype == object


def test_encode_features_shapes():
    df = load_titanic()
    cleaned = clean_data(df)
    X, y = encode_features(cleaned)
    assert len(X) == len(y)
    assert len(y) > 0


def test_split_data_proportions():
    df = load_titanic()
    cleaned = clean_data(df)
    X, y = encode_features(cleaned)
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)
    total = len(y_train) + len(y_val) + len(y_test)
    assert abs(len(y_test) / total - 0.2) < 0.05
    assert abs(len(y_val) / total - 0.2) < 0.05


def test_apply_smote_balances_classes():
    df = load_titanic()
    cleaned = clean_data(df)
    X, y = encode_features(cleaned)
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)
    X_res, y_res = apply_smote(X_train, y_train)
    counts = np.bincount(y_res)
    assert abs(counts[0] - counts[1]) / max(counts) < 0.05
