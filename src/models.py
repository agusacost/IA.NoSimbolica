from sklearn.neural_network import MLPClassifier


MODEL_CONFIGS = [
    {
        "name": "Modelo 1 – Baseline MLP",
        "description": "1 capa oculta (64 neuronas), ReLU, SGD, lr=0.01, 100 épocas",
        "params": {
            "hidden_layer_sizes": (64,),
            "activation": "relu",
            "solver": "sgd",
            "learning_rate_init": 0.01,
            "max_iter": 100,
            "random_state": 42,
            "verbose": False,
        },
    },
    {
        "name": "Modelo 2 – MLP Profundo",
        "description": "3 capas ocultas (128-64-32 neuronas), ReLU, Adam, lr=0.001, 200 épocas",
        "params": {
            "hidden_layer_sizes": (128, 64, 32),
            "activation": "relu",
            "solver": "adam",
            "learning_rate_init": 0.001,
            "max_iter": 200,
            "random_state": 42,
            "verbose": False,
        },
    },
    {
        "name": "Modelo 3 – MLP Regularizado",
        "description": "2 capas ocultas (64-32 neuronas), tanh, Adam, lr=0.001, alpha=0.01, 200 épocas",
        "params": {
            "hidden_layer_sizes": (64, 32),
            "activation": "tanh",
            "solver": "adam",
            "learning_rate_init": 0.001,
            "max_iter": 200,
            "alpha": 0.01,
            "random_state": 42,
            "verbose": False,
        },
    },
]


def build_model_1() -> MLPClassifier:
    return MLPClassifier(**MODEL_CONFIGS[0]["params"])


def build_model_2() -> MLPClassifier:
    return MLPClassifier(**MODEL_CONFIGS[1]["params"])


def build_model_3() -> MLPClassifier:
    return MLPClassifier(**MODEL_CONFIGS[2]["params"])
