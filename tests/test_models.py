import numpy as np
from sklearn.neural_network import MLPClassifier
from src.models import build_model_1, build_model_2, build_model_3, MODEL_CONFIGS


def test_build_model_1_returns_mlp():
    model = build_model_1()
    assert isinstance(model, MLPClassifier)


def test_build_model_2_returns_mlp():
    model = build_model_2()
    assert isinstance(model, MLPClassifier)


def test_build_model_3_returns_mlp():
    model = build_model_3()
    assert isinstance(model, MLPClassifier)


def test_model_configs_has_three_entries():
    assert len(MODEL_CONFIGS) == 3


def test_model_1_architecture():
    model = build_model_1()
    assert model.hidden_layer_sizes == (64,)
    assert model.activation == "relu"
    assert model.solver == "sgd"


def test_model_2_architecture():
    model = build_model_2()
    assert model.hidden_layer_sizes == (128, 64, 32)
    assert model.activation == "relu"
    assert model.solver == "adam"


def test_model_3_architecture():
    model = build_model_3()
    assert model.hidden_layer_sizes == (64, 32)
    assert model.activation == "tanh"
    assert model.solver == "adam"


def test_models_can_fit_dummy_data():
    X = np.random.rand(100, 8)
    y = np.random.randint(0, 2, 100)
    for builder in [build_model_1, build_model_2, build_model_3]:
        model = builder()
        model.fit(X, y)
        preds = model.predict(X)
        assert len(preds) == 100
