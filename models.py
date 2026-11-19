# models.py
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier


# ---------- Factory: tanda tangan seragam (params, random_state) ----------

def _logistic(params, rs):
    return LogisticRegression(
        C=params.get("C", 1.0), max_iter=2000, random_state=rs,
    )

def _decision_tree(params, rs):
    return DecisionTreeClassifier(
        max_depth=params.get("max_depth", 10),
        min_samples_split=params.get("min_samples_split", 2),
        random_state=rs,
    )

def _random_forest(params, rs):
    return RandomForestClassifier(
        n_estimators=params.get("n_estimators", 200),
        max_depth=params.get("max_depth", None),
        random_state=rs, n_jobs=-1,
    )

def _gradient_boosting(params, rs):
    return GradientBoostingClassifier(
        n_estimators=params.get("n_estimators", 100),
        learning_rate=params.get("learning_rate", 0.1),
        max_depth=params.get("max_depth", 3),
        random_state=rs,
    )

def _zero_r(params, rs):
    return DummyClassifier(strategy="most_frequent")


# ---------- Registry ----------

MODELS = {
    "zero_r": {
        "label": "Zero-R (Baseline)",
        "factory": _zero_r,
        "supports_class_weight": False,
    },
    "logistic": {
        "label": "Logistic Regression",
        "factory": _logistic,
        "supports_class_weight": True,
    },
    "decision_tree": {
        "label": "Decision Tree",
        "factory": _decision_tree,
        "supports_class_weight": True,
    },
    "random_forest": {
        "label": "Random Forest",
        "factory": _random_forest,
        "supports_class_weight": True,
    },
    "gradient_boosting": {
        "label": "Gradient Boosting",
        "factory": _gradient_boosting,
        "supports_class_weight": False,
    },
}


# ---------- Spesifikasi parameter sidebar ----------

PARAM_SPECS = {
    "logistic": [
        {"name": "C", "label": "C (inverse regularisasi)", "type": "float_log",
         "options": [0.01, 0.1, 1.0, 10.0, 100.0], "default": 1.0},
    ],
    "decision_tree": [
        {"name": "max_depth", "label": "max_depth", "type": "int",
         "min": 2, "max": 40, "default": 10},
        {"name": "min_samples_split", "label": "min_samples_split", "type": "int",
         "min": 2, "max": 30, "default": 2},
    ],
    "random_forest": [
        {"name": "n_estimators", "label": "n_estimators", "type": "int",
         "min": 50, "max": 600, "default": 200, "step": 50},
        {"name": "max_depth", "label": "max_depth (0 = None)", "type": "int",
         "min": 0, "max": 40, "default": 0},
    ],
    "gradient_boosting": [
        {"name": "n_estimators", "label": "n_estimators", "type": "int",
         "min": 50, "max": 500, "default": 100, "step": 50},
        {"name": "learning_rate", "label": "learning_rate", "type": "float_log",
         "options": [0.01, 0.05, 0.1, 0.2, 0.3], "default": 0.1},
        {"name": "max_depth", "label": "max_depth", "type": "int",
         "min": 1, "max": 8, "default": 3},
    ],
    "zero_r": [],
}


# ---------- Grid kecil untuk tuning (dipakai evaluasi) ----------

GRIDS = {
    "logistic": {"classifier__C": [0.1, 1.0, 10.0]},
    "decision_tree": {
        "classifier__max_depth": [5, 10, 20, None],
        "classifier__min_samples_split": [2, 10],
    },
    "random_forest": {
        "classifier__n_estimators": [100, 300],
        "classifier__max_depth": [None, 15],
    },
    "gradient_boosting": {
        "classifier__n_estimators": [100, 200],
        "classifier__learning_rate": [0.05, 0.1],
    },
    "zero_r": {},
}


def build_estimator(model_key, params, random_state, class_weight=None):
    spec = MODELS[model_key]
    est = spec["factory"](params or {}, random_state)

    if class_weight and spec.get("supports_class_weight", False):
        est.set_params(class_weight=class_weight)

    # max_depth = 0 di UI berarti None
    if model_key == "random_forest" and (params or {}).get("max_depth", 1) == 0:
        est.set_params(max_depth=None)

    return est