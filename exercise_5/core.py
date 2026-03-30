"""Exercise 5 classification workflows."""

from __future__ import annotations

import logging
import os
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn.ensemble
import sklearn.metrics as metrics
from matplotlib.colors import Normalize
from sklearn.datasets import fetch_openml
from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, cross_val_predict
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelBinarizer, LabelEncoder, RobustScaler

log = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent
STEP_1_DIR = BASE_DIR / "step_1"
STEP_2_DIR = BASE_DIR / "step_2"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _plot_roc(
    ax: plt.Axes,
    fpr: np.ndarray,
    tpr: np.ndarray,
    roc_auc: float,
    *,
    color: str = "b",
    name: str = "",
) -> None:
    """Plot a single ROC curve on the given axes."""
    ax.set_title("Receiver Operating Characteristic")
    ax.plot(fpr, tpr, color, label=f"{name} AUC = {roc_auc:.4f}")
    ax.legend(loc="lower right")
    ax.plot([0, 1], [0, 1], "r--")
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.set_ylabel("True Positive Rate")
    ax.set_xlabel("False Positive Rate")


# ---------------------------------------------------------------------------
# Exercise 5.1 — MNIST classification
# ---------------------------------------------------------------------------

def exercise_5_1() -> None:
    """MNIST digit classification: binary (is-5) and multiclass (0-9)."""
    images_dir = STEP_1_DIR / "images"
    _ensure_dir(images_dir)

    print("[5.1] Downloading MNIST dataset (cached after first run)...")
    mnist = fetch_openml("mnist_784", version=1, as_frame=False, parser="auto")
    X: np.ndarray = mnist["data"]
    y: np.ndarray = mnist["target"]

    X_train, X_test = X[:60000], X[60000:]
    y_train, y_test = y[:60000], y[60000:]

    # ---- Binary classification: detect digit 5 ----
    print("[5.1] Binary classification — is-5 detector")
    y_train_5 = (y_train == "5")
    y_test_5 = (y_test == "5")

    sgd_clf = SGDClassifier(random_state=42)
    y_train_pred = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3)

    cm = confusion_matrix(y_train_5, y_train_pred)
    prec = precision_score(y_train_5, y_train_pred)
    rec = recall_score(y_train_5, y_train_pred)
    f1 = f1_score(y_train_5, y_train_pred)

    print(f"  Confusion matrix:\n{cm}")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall:    {rec:.4f}")
    print(f"  F1 score:  {f1:.4f}")

    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay.from_predictions(y_train_5, y_train_pred, ax=ax)
    ax.set_title("Binary Confusion Matrix (is-5 detector)")
    fig.tight_layout()
    fig.savefig(images_dir / "binary_confusion_matrix.png", dpi=150)
    plt.close(fig)
    print(f"  Saved {images_dir / 'binary_confusion_matrix.png'}")

    # ROC curve
    y_scores = cross_val_predict(
        sgd_clf, X_train, y_train_5, cv=3, method="decision_function",
    )
    fpr, tpr, _ = roc_curve(y_train_5, y_scores)
    auc_score = roc_auc_score(y_train_5, y_scores)
    print(f"  ROC AUC: {auc_score:.4f}")

    fig, ax = plt.subplots(figsize=(6, 5))
    _plot_roc(ax, fpr, tpr, auc_score, color="b", name="SGD is-5")
    fig.tight_layout()
    fig.savefig(images_dir / "binary_roc_curve.png", dpi=150)
    plt.close(fig)
    print(f"  Saved {images_dir / 'binary_roc_curve.png'}")

    # ---- Multiclass classification ----
    print("[5.1] Multiclass classification — 10 digits")
    sgd_clf_multi = SGDClassifier(random_state=42)
    y_train_pred_multi = cross_val_predict(sgd_clf_multi, X_train, y_train, cv=3)

    accuracy = np.mean(y_train_pred_multi == y_train)
    print(f"  Overall accuracy: {accuracy:.4f}")

    cm_multi = confusion_matrix(y_train, y_train_pred_multi)
    fig, ax = plt.subplots(figsize=(8, 7))
    ConfusionMatrixDisplay.from_predictions(
        y_train, y_train_pred_multi, ax=ax, cmap="Blues",
    )
    ax.set_title("Multiclass Confusion Matrix (SGD, 10 digits)")
    fig.tight_layout()
    fig.savefig(images_dir / "multiclass_confusion_matrix.png", dpi=150)
    plt.close(fig)
    print(f"  Saved {images_dir / 'multiclass_confusion_matrix.png'}")

    # Normalized error matrix (diagonal zeroed)
    row_sums = cm_multi.sum(axis=1, keepdims=True)
    norm_cm = cm_multi / row_sums
    np.fill_diagonal(norm_cm, 0)

    fig, ax = plt.subplots(figsize=(8, 7))
    ax.matshow(norm_cm, cmap="Reds")
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title("Confusion Errors (normalized, diagonal zeroed)")
    ax.set_xticks(range(10))
    ax.set_yticks(range(10))
    fig.tight_layout()
    fig.savefig(images_dir / "multiclass_confusion_errors.png", dpi=150)
    plt.close(fig)
    print(f"  Saved {images_dir / 'multiclass_confusion_errors.png'}")


# ---------------------------------------------------------------------------
# Exercise 5.2 — Bank Customer Churn
# ---------------------------------------------------------------------------

def exercise_5_2() -> None:
    """Bank Customer Churn: feature engineering + GridSearchCV to beat AutoML."""
    images_dir = STEP_2_DIR / "images"
    _ensure_dir(images_dir)

    csv_path = STEP_2_DIR / "datasets" / "Churn_Modelling.csv"
    if not csv_path.is_file():
        raise FileNotFoundError(
            f"Dataset not found at {csv_path}. "
            "Ensure Churn_Modelling.csv is in exercise_5/step_2/datasets/."
        )

    data = pd.read_csv(csv_path)
    random_seed = 0
    np.random.seed(random_seed)
    msk = np.random.rand(len(data)) < 0.8
    train = data[msk].reset_index(drop=True)
    validation = data[~msk].reset_index(drop=True)

    # ---- Baseline Random Forest ----
    print("[5.2] Baseline Random Forest (max_depth=6)")
    drop_cols = ["Exited", "RowNumber", "CustomerId", "Surname"]
    t_rf_base = train.drop(drop_cols, axis=1).copy()
    v_rf_base = validation.drop(drop_cols, axis=1).copy()

    t_rf_base["Gender"] = t_rf_base["Gender"].replace({"Female": 1, "Male": 0})
    v_rf_base["Gender"] = v_rf_base["Gender"].replace({"Female": 1, "Male": 0})

    le_geo = LabelEncoder()
    le_geo.fit(t_rf_base["Geography"])
    t_rf_base["Geography"] = le_geo.transform(t_rf_base["Geography"])
    v_rf_base["Geography"] = le_geo.transform(v_rf_base["Geography"])

    clf_base = RandomForestClassifier(max_depth=6, random_state=123)
    clf_base.fit(t_rf_base, train["Exited"])
    preds_base = clf_base.predict_proba(v_rf_base)[:, 1]

    fpr_base, tpr_base, thresh_base = roc_curve(validation["Exited"], preds_base)
    auc_base = metrics.auc(fpr_base, tpr_base)
    print(f"  Baseline RF AUC: {auc_base:.4f}")

    # ---- Feature engineering ----
    print("[5.2] Feature engineering (LabelBinarizer, RobustScaler, VarianceThreshold)")
    lb_surname = LabelBinarizer()
    lb_surname.fit(train["Surname"].unique())
    surname_train = pd.DataFrame(
        lb_surname.transform(train["Surname"]),
        columns=lb_surname.classes_,
    )
    surname_val = pd.DataFrame(
        lb_surname.transform(validation["Surname"]),
        columns=lb_surname.classes_,
    )

    lb_geo = LabelBinarizer()
    lb_geo.fit(train["Geography"].unique())
    geo_train = pd.DataFrame(
        lb_geo.transform(train["Geography"]),
        columns=lb_geo.classes_,
    )
    geo_val = pd.DataFrame(
        lb_geo.transform(validation["Geography"]),
        columns=lb_geo.classes_,
    )

    drop_extra = ["Exited", "RowNumber", "CustomerId", "Surname", "Geography"]
    t_rf = pd.concat(
        [train.drop(drop_extra, axis=1), surname_train, geo_train], axis=1,
    )
    v_rf = pd.concat(
        [validation.drop(drop_extra, axis=1), surname_val, geo_val], axis=1,
    )
    t_rf["Gender"] = t_rf["Gender"].replace({"Female": 1, "Male": 0})
    v_rf["Gender"] = v_rf["Gender"].replace({"Female": 1, "Male": 0})

    scaler = RobustScaler().fit(t_rf)
    t_rf_scaled = scaler.transform(t_rf)
    v_rf_scaled = scaler.transform(v_rf)

    selector = VarianceThreshold(threshold=0.001).fit(t_rf_scaled)
    t_rf_selected = selector.transform(t_rf_scaled)
    v_rf_selected = selector.transform(v_rf_scaled)

    # ---- GridSearchCV ----
    cache_path = STEP_2_DIR / "grid_search_cache.pkl"
    random_state = 123

    if not cache_path.is_file():
        print("[5.2] Running GridSearchCV (this may take several minutes)...")
        pipe = Pipeline([("Classifier", RandomForestClassifier())])
        params = [
            {
                "Classifier": [RandomForestClassifier(random_state=random_state)],
                "Classifier__criterion": ["entropy", "gini", "log_loss"],
                "Classifier__n_estimators": [70, 100, 120, 170, 200, 300],
                "Classifier__max_depth": [None, 5, 10, 15],
            },
            {
                "Classifier": [GradientBoostingClassifier(random_state=random_state)],
                "Classifier__loss": ["log_loss", "exponential"],
                "Classifier__n_estimators": [70, 100, 120, 170, 200, 300],
                "Classifier__criterion": ["friedman_mse", "squared_error"],
            },
            {
                "Classifier": [
                    AdaBoostClassifier(random_state=random_state, algorithm="SAMME"),
                ],
                "Classifier__n_estimators": [20, 50, 70, 100, 120, 170],
            },
            {
                "Classifier": [KNeighborsClassifier()],
                "Classifier__n_neighbors": [3, 5, 7, 15, 21],
                "Classifier__weights": ["uniform", "distance"],
            },
        ]
        grid_search = GridSearchCV(
            pipe, params, scoring="roc_auc", cv=3, refit=True, n_jobs=-1,
        )
        grid_search.fit(t_rf_selected, train["Exited"])
        with open(cache_path, "wb") as fh:
            pickle.dump(grid_search, fh)
        print("  GridSearchCV cached.")
    else:
        print("[5.2] Loading cached GridSearchCV result...")
        with open(cache_path, "rb") as fh:
            grid_search = pickle.load(fh)

    print(f"  Best params: {grid_search.best_params_}")

    preds_top = grid_search.predict_proba(v_rf_selected)[:, 1]
    fpr_top, tpr_top, thresh_top = roc_curve(validation["Exited"], preds_top)
    auc_top = metrics.auc(fpr_top, tpr_top)
    print(f"  GridSearch best AUC: {auc_top:.4f}")

    # ---- ROC comparison plot ----
    fig, ax = plt.subplots(figsize=(7, 6))
    _plot_roc(ax, fpr_base, tpr_base, auc_base, color="b", name="Baseline RF")
    _plot_roc(ax, fpr_top, tpr_top, auc_top, color="k", name="GridSearch best")
    fig.tight_layout()
    fig.savefig(images_dir / "roc_comparison.png", dpi=150)
    plt.close(fig)
    print(f"  Saved {images_dir / 'roc_comparison.png'}")

    # ---- Evaluation at 80% TPR threshold ----
    _evaluate_at_tpr(
        "Baseline RF",
        validation["Exited"],
        preds_base,
        tpr_base,
        fpr_base,
        thresh_base,
        images_dir / "confusion_matrix_baseline.png",
    )
    _evaluate_at_tpr(
        "GridSearch best",
        validation["Exited"],
        preds_top,
        tpr_top,
        fpr_top,
        thresh_top,
        images_dir / "confusion_matrix_gridsearch.png",
    )


def _evaluate_at_tpr(
    name: str,
    y_true: pd.Series,
    preds: np.ndarray,
    tpr: np.ndarray,
    fpr: np.ndarray,
    thresholds: np.ndarray,
    save_path: Path,
    *,
    target_tpr: float = 0.8,
) -> None:
    """Evaluate a classifier at a target TPR threshold and save confusion matrix."""
    target_idx = int(np.min(np.where(tpr > target_tpr)[0]))
    target_thresh = thresholds[target_idx]
    binary_preds = (preds > target_thresh).astype(int)
    acc = metrics.accuracy_score(y_true, binary_preds)

    print(f"  [{name}] at TPR > {target_tpr:.0%}:")
    print(f"    FPR = {fpr[target_idx]:.3f}")
    print(f"    Threshold = {target_thresh:.3f}")
    print(f"    Accuracy = {acc:.4f}")

    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay.from_predictions(y_true, binary_preds, ax=ax)
    ax.set_title(f"Confusion Matrix {name} | Accuracy = {acc:.3f}")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"    Saved {save_path}")


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

EXERCISE_HANDLERS: dict[str, callable] = {
    "5_1": exercise_5_1,
    "5_2": exercise_5_2,
}


def run_exercises(exercise_ids: list[str]) -> None:
    """Run the requested exercise steps in order."""
    for exercise_id in exercise_ids:
        print(f"\n{'=' * 60}")
        print(f"[Runner] Running exercise {exercise_id}")
        print(f"{'=' * 60}")
        EXERCISE_HANDLERS[exercise_id]()
