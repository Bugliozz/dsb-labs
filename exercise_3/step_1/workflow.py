"""Offline training workflow for exercise 3.1."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

try:
    from .ames import (
        ARTIFACT_DIR,
        DATA_DIR,
        DEMO_FEATURE_ORDER,
        IMAGE_DIR,
        build_input_defaults,
        build_input_schema,
        evaluate_holdout,
        load_training_frame,
        prepare_raw_training_data,
        run_model_search,
        save_artifacts,
        split_training_data,
        utc_timestamp,
        validate_dataset_files,
    )
except ImportError:  # pragma: no cover - fallback for step-local execution
    from ames import (  # type: ignore
        ARTIFACT_DIR,
        DATA_DIR,
        DEMO_FEATURE_ORDER,
        IMAGE_DIR,
        build_input_defaults,
        build_input_schema,
        evaluate_holdout,
        load_training_frame,
        prepare_raw_training_data,
        run_model_search,
        save_artifacts,
        split_training_data,
        utc_timestamp,
        validate_dataset_files,
    )


def run_training_workflow(
    data_dir: Path | None = None,
    image_dir: Path | None = None,
    artifact_dir: Path | None = None,
) -> dict[str, Any]:
    """Run the full Ames training workflow and persist inference artifacts."""
    data_dir = Path(data_dir) if data_dir else DATA_DIR
    image_dir = Path(image_dir) if image_dir else IMAGE_DIR
    artifact_dir = Path(artifact_dir) if artifact_dir else ARTIFACT_DIR

    train_csv, description = validate_dataset_files(data_dir)
    frame = load_training_frame(data_dir)
    X, y = prepare_raw_training_data(frame)
    input_defaults = build_input_defaults(X)
    input_schema = build_input_schema(X)

    image_dir.mkdir(parents=True, exist_ok=True)
    create_eda_outputs(frame, image_dir)

    X_train, X_test, y_train, y_test = split_training_data(X, y)
    best_model, metrics, run_summaries, mutual_info_report = run_model_search(X_train, y_train, X_test, y_test)
    selected_holdout = evaluate_holdout(best_model, X_test, y_test)
    model_info = {
        "training_timestamp": utc_timestamp(),
        "selected_model_name": metrics["selected_model"]["name"],
        "selected_model_params": metrics["selected_model"]["best_params"],
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "raw_feature_count": int(X.shape[1]),
        "demo_feature_order": DEMO_FEATURE_ORDER,
        "dataset_path": str(train_csv),
        "data_description_path": str(description),
        "artifacts_path": str(artifact_dir),
        "images_path": str(image_dir),
        "holdout_metrics": selected_holdout,
    }
    save_artifacts(
        model=best_model,
        metrics=metrics,
        model_info=model_info,
        input_schema=input_schema,
        input_defaults=input_defaults,
        mutual_info_report=mutual_info_report,
        artifact_dir=artifact_dir,
    )

    print("[Exercise 3.1] Ames training workflow completed.")
    print(f"[Exercise 3.1] dataset: {train_csv}")
    print(f"[Exercise 3.1] description: {description}")
    print(f"[Exercise 3.1] images saved to: {image_dir}")
    print(f"[Exercise 3.1] artifacts saved to: {artifact_dir}")
    print(
        "[Exercise 3.1] selected model: "
        f"{metrics['selected_model']['name']} (CV RMSE={metrics['selected_model']['cv_rmse']:.2f})"
    )
    print(
        "[Exercise 3.1] holdout metrics: "
        f"RMSE={selected_holdout['rmse']:.2f}, "
        f"MAE={selected_holdout['mae']:.2f}, "
        f"R2={selected_holdout['r2']:.4f}"
    )
    print("[Exercise 3.1] next steps:")
    print("  - Train again: python exercise_3/main.py train")
    print("  - Local app: python exercise_3/main.py serve")
    print("  - Docker build: docker build -t ames-house-price-api exercise_3/step_1")
    print("  - Docker run: docker run --rm -p 5000:5000 ames-house-price-api")

    return {
        "train_csv": train_csv,
        "description": description,
        "image_dir": image_dir,
        "artifact_dir": artifact_dir,
        "metrics": metrics,
        "model_info": model_info,
        "runs": run_summaries,
    }


def create_eda_outputs(frame: pd.DataFrame, image_dir: Path) -> None:
    """Create the EDA plots requested by the exercise."""
    sns.set_theme(style="whitegrid")

    sale_price_path = image_dir / "sale_price_distribution.png"
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(frame["SalePrice"], kde=True, ax=ax, color="#0b7285")
    ax.set_title("SalePrice distribution")
    ax.set_xlabel("SalePrice")
    fig.tight_layout()
    fig.savefig(sale_price_path, dpi=150)
    plt.close(fig)

    missing_share = frame.isna().mean().sort_values(ascending=False).head(20)
    missing_path = image_dir / "missing_values_top20.png"
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x=missing_share.values, y=missing_share.index, ax=ax, color="#d94841")
    ax.set_title("Top 20 missing-value ratios")
    ax.set_xlabel("Missing share")
    ax.set_ylabel("Feature")
    fig.tight_layout()
    fig.savefig(missing_path, dpi=150)
    plt.close(fig)

    numeric_frame = frame.select_dtypes(include=["number"])
    top_corr = numeric_frame.corr(numeric_only=True)["SalePrice"].abs().sort_values(ascending=False).head(12).index
    corr_path = image_dir / "top_numeric_correlations.png"
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(numeric_frame[top_corr].corr(), annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
    ax.set_title("Top numeric correlations with SalePrice")
    fig.tight_layout()
    fig.savefig(corr_path, dpi=150)
    plt.close(fig)

    skewness = numeric_frame.drop(columns=["SalePrice"], errors="ignore").skew().abs().sort_values(ascending=False).head(8)
    skewed_columns = list(skewness.index)
    if skewed_columns:
        boxplot_path = image_dir / "skewed_numeric_boxplots.png"
        melted = frame[skewed_columns].melt(var_name="feature", value_name="value")
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.boxplot(data=melted, x="feature", y="value", ax=ax, color="#74c0fc")
        ax.set_title("Most skewed numeric features")
        ax.tick_params(axis="x", rotation=45)
        fig.tight_layout()
        fig.savefig(boxplot_path, dpi=150)
        plt.close(fig)
