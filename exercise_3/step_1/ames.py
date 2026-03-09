"""Shared Ames housing helpers for training and inference."""

from __future__ import annotations

import ast
import inspect
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.feature_selection import VarianceThreshold, mutual_info_regression
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold, train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
IMAGE_DIR = BASE_DIR / "images"
ARTIFACT_DIR = BASE_DIR / "artifacts"
TRAIN_CSV = DATA_DIR / "train.csv"
DATA_DESCRIPTION = DATA_DIR / "data_description.txt"
BEST_MODEL_PATH = ARTIFACT_DIR / "best_model.joblib"
METRICS_PATH = ARTIFACT_DIR / "metrics.json"
MODEL_INFO_PATH = ARTIFACT_DIR / "model_info.json"
INPUT_SCHEMA_PATH = ARTIFACT_DIR / "input_schema.json"
INPUT_DEFAULTS_PATH = ARTIFACT_DIR / "input_defaults.json"
MUTUAL_INFO_PATH = ARTIFACT_DIR / "mutual_information.csv"
DEMO_FEATURE_ORDER = ["LotArea", "YearBuilt", "GrLivArea", "GarageCars", "OpenPorchSF"]
ENGINEERED_FEATURES = ["HouseAge", "RemodelAge", "TotalBath", "TotalSF"]
RANDOM_STATE = 42


class AmesFeatureEngineer(BaseEstimator, TransformerMixin):
    """Create additional numeric features from the raw Ames columns."""

    source_columns = (
        "YrSold",
        "YearBuilt",
        "YearRemodAdd",
        "BsmtFullBath",
        "BsmtHalfBath",
        "FullBath",
        "HalfBath",
        "TotalBsmtSF",
        "1stFlrSF",
        "2ndFlrSF",
    )

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        frame = _ensure_dataframe(X).copy()
        numeric: dict[str, pd.Series] = {}
        for column in self.source_columns:
            if column in frame.columns:
                numeric[column] = pd.to_numeric(frame[column], errors="coerce")
            else:
                numeric[column] = pd.Series(np.nan, index=frame.index, dtype="float64")

        frame["HouseAge"] = numeric["YrSold"] - numeric["YearBuilt"]
        frame["RemodelAge"] = numeric["YrSold"] - numeric["YearRemodAdd"]
        frame["TotalBath"] = (
            numeric["BsmtFullBath"].fillna(0)
            + 0.5 * numeric["BsmtHalfBath"].fillna(0)
            + numeric["FullBath"].fillna(0)
            + 0.5 * numeric["HalfBath"].fillna(0)
        )
        frame["TotalSF"] = (
            numeric["TotalBsmtSF"].fillna(0)
            + numeric["1stFlrSF"].fillna(0)
            + numeric["2ndFlrSF"].fillna(0)
        )
        return frame


class IQRCapper(BaseEstimator, TransformerMixin):
    """Cap numeric values using the interquartile range."""

    def fit(self, X, y=None):
        frame = _ensure_dataframe(X)
        numeric = frame.apply(pd.to_numeric, errors="coerce")
        q1 = numeric.quantile(0.25)
        q3 = numeric.quantile(0.75)
        iqr = q3 - q1
        self.columns_ = list(numeric.columns)
        self.lower_bounds_ = (q1 - 1.5 * iqr).fillna(q1)
        self.upper_bounds_ = (q3 + 1.5 * iqr).fillna(q3)
        return self

    def transform(self, X):
        frame = _ensure_dataframe(X, columns=getattr(self, "columns_", None))
        numeric = frame.apply(pd.to_numeric, errors="coerce")
        return numeric.clip(self.lower_bounds_, self.upper_bounds_, axis=1)


def _ensure_dataframe(data, columns: list[str] | None = None) -> pd.DataFrame:
    """Return a DataFrame preserving column names when possible."""
    if isinstance(data, pd.DataFrame):
        return data.copy()
    if columns is None:
        return pd.DataFrame(data)
    return pd.DataFrame(data, columns=columns)


def utc_timestamp() -> str:
    """Return the current timestamp in ISO 8601 UTC format."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def make_one_hot_encoder() -> OneHotEncoder:
    """Build an OneHotEncoder compatible with multiple sklearn versions."""
    kwargs: dict[str, Any] = {"handle_unknown": "ignore"}
    if "sparse_output" in inspect.signature(OneHotEncoder).parameters:
        kwargs["sparse_output"] = False
    else:
        kwargs["sparse"] = False
    return OneHotEncoder(**kwargs)


def validate_dataset_files(data_dir: Path | None = None) -> tuple[Path, Path]:
    """Ensure the required Ames files exist before training."""
    data_dir = Path(data_dir) if data_dir else DATA_DIR
    train_csv = data_dir / "train.csv"
    description = data_dir / "data_description.txt"
    missing = [path.name for path in (train_csv, description) if not path.is_file()]
    if missing:
        missing_joined = ", ".join(missing)
        raise FileNotFoundError(
            f"Missing Ames dataset files in {data_dir}: {missing_joined}. "
            "Place the Kaggle train.csv and data_description.txt files there."
        )
    return train_csv, description


def load_training_frame(data_dir: Path | None = None) -> pd.DataFrame:
    """Load the Ames train split used for the regression workflow."""
    train_csv, _ = validate_dataset_files(data_dir)
    frame = pd.read_csv(train_csv)
    if "SalePrice" not in frame.columns:
        raise ValueError("train.csv must include the `SalePrice` target column.")
    return frame


def prepare_raw_training_data(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Drop target and identifier columns and return X/y for modeling."""
    data = frame.copy()
    y = data.pop("SalePrice")
    if "Id" in data.columns:
        data = data.drop(columns=["Id"])
    return data, y


def build_stratification_bins(target: pd.Series, max_bins: int = 10) -> pd.Series:
    """Create quantile bins for stratified regression splitting."""
    quantiles = min(max_bins, max(2, target.nunique()))
    bins = pd.qcut(target, q=quantiles, duplicates="drop")
    if bins.nunique() < 2:
        return pd.cut(target, bins=2, labels=False, include_lowest=True)
    return bins.cat.codes


def split_training_data(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split Ames data into train and test sets using stratified quantile bins."""
    strat_bins = build_stratification_bins(y)
    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=strat_bins,
    )


def build_input_defaults(raw_frame: pd.DataFrame) -> dict[str, Any]:
    """Build fallback defaults for all raw Ames input fields."""
    defaults: dict[str, Any] = {}
    for column in raw_frame.columns:
        series = raw_frame[column]
        if pd.api.types.is_numeric_dtype(series):
            median = pd.to_numeric(series, errors="coerce").median()
            defaults[column] = float(median) if pd.notna(median) else 0.0
        else:
            mode = series.dropna().astype(str).mode()
            defaults[column] = mode.iloc[0] if not mode.empty else ""
    return defaults


def build_input_schema(raw_frame: pd.DataFrame) -> dict[str, Any]:
    """Describe the raw Ames columns accepted by the API."""
    fields: dict[str, Any] = {}
    numeric_features: list[str] = []
    categorical_features: list[str] = []

    for column in raw_frame.columns:
        series = raw_frame[column]
        if pd.api.types.is_numeric_dtype(series):
            numeric_features.append(column)
            fields[column] = {"type": "number"}
        else:
            categorical_features.append(column)
            categories = sorted({str(value) for value in series.dropna().astype(str).tolist()})
            fields[column] = {"type": "categorical", "allowed_values": categories}

    return {
        "feature_order": list(raw_frame.columns),
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "demo_feature_order": DEMO_FEATURE_ORDER,
        "engineered_features": ENGINEERED_FEATURES,
        "fields": fields,
    }


def build_model_specs() -> list[dict[str, Any]]:
    """Return the fixed grid-search configuration for all candidate regressors."""
    return [
        {
            "name": "LinearRegression",
            "estimator": LinearRegression(),
            "param_grid": [{}],
        },
        {
            "name": "Ridge",
            "estimator": Ridge(),
            "param_grid": [{"model__alpha": [0.001, 0.01, 0.1, 1, 10]}],
        },
        {
            "name": "Lasso",
            "estimator": Lasso(max_iter=20000),
            "param_grid": [{"model__alpha": [0.001, 0.01, 0.1, 1, 10]}],
        },
        {
            "name": "KNeighborsRegressor",
            "estimator": KNeighborsRegressor(),
            "param_grid": [
                {
                    "model__n_neighbors": [3, 5, 7, 11],
                    "model__weights": ["uniform", "distance"],
                }
            ],
        },
        {
            "name": "DecisionTreeRegressor",
            "estimator": DecisionTreeRegressor(random_state=RANDOM_STATE),
            "param_grid": [
                {
                    "model__max_depth": [None, 10, 20, 40],
                    "model__min_samples_leaf": [1, 2, 4],
                }
            ],
        },
        {
            "name": "RandomForestRegressor",
            "estimator": RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=1),
            "param_grid": [
                {
                    "model__n_estimators": [200, 400],
                    "model__max_depth": [None, 20, 40],
                    "model__min_samples_leaf": [1, 2, 4],
                }
            ],
        },
        {
            "name": "GradientBoostingRegressor",
            "estimator": GradientBoostingRegressor(random_state=RANDOM_STATE),
            "param_grid": [
                {
                    "model__n_estimators": [200, 400],
                    "model__learning_rate": [0.03, 0.05, 0.1],
                    "model__max_depth": [2, 3],
                    "model__subsample": [0.8, 1.0],
                }
            ],
        },
    ]


def build_model_pipeline(raw_feature_frame: pd.DataFrame, estimator: BaseEstimator) -> Pipeline:
    """Build the end-to-end regression pipeline starting from raw Ames columns."""
    engineer = AmesFeatureEngineer()
    engineered = engineer.transform(raw_feature_frame)
    numeric_features = engineered.select_dtypes(include=["number"]).columns.tolist()
    categorical_features = [column for column in engineered.columns if column not in numeric_features]

    numeric_pipeline = Pipeline(
        steps=[
            ("iqr_capper", IQRCapper()),
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", make_one_hot_encoder()),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_features),
            ("cat", categorical_pipeline, categorical_features),
        ],
        remainder="drop",
        sparse_threshold=0.0,
    )
    return Pipeline(
        steps=[
            ("feature_engineering", engineer),
            ("preprocessor", preprocessor),
            ("variance", VarianceThreshold()),
            ("model", estimator),
        ]
    )


def evaluate_holdout(model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> dict[str, float]:
    """Compute holdout regression metrics for a fitted model."""
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    return {
        "mse": float(mse),
        "rmse": float(np.sqrt(mse)),
        "mae": float(mean_absolute_error(y_test, predictions)),
        "r2": float(r2_score(y_test, predictions)),
    }


def run_model_search(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> tuple[Pipeline, dict[str, Any], list[dict[str, Any]], pd.DataFrame]:
    """Train all candidate models and return the best estimator and reports."""
    cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    runs: list[dict[str, Any]] = []
    best_entry: dict[str, Any] | None = None

    for spec in build_model_specs():
        pipeline = build_model_pipeline(X_train, spec["estimator"])
        search = GridSearchCV(
            estimator=pipeline,
            param_grid=spec["param_grid"],
            scoring="neg_root_mean_squared_error",
            cv=cv,
            n_jobs=1,
            refit=True,
        )
        search.fit(X_train, y_train)
        result = {
            "model_name": spec["name"],
            "best_params": search.best_params_,
            "cv_rmse": float(-search.best_score_),
            "holdout": evaluate_holdout(search.best_estimator_, X_test, y_test),
            "estimator": search.best_estimator_,
        }
        runs.append(result)
        if best_entry is None or result["cv_rmse"] < best_entry["cv_rmse"]:
            best_entry = result

    if best_entry is None:
        raise RuntimeError("No model candidates were evaluated.")

    baseline_entry = next(run for run in runs if run["model_name"] == "LinearRegression")
    metrics = {
        "generated_at": utc_timestamp(),
        "selected_model": {
            "name": best_entry["model_name"],
            "best_params": best_entry["best_params"],
            "cv_rmse": best_entry["cv_rmse"],
            "holdout": best_entry["holdout"],
        },
        "baseline_linear_regression": {
            "cv_rmse": baseline_entry["cv_rmse"],
            "holdout": baseline_entry["holdout"],
        },
        "selected_beats_linear_holdout_rmse": bool(
            best_entry["holdout"]["rmse"] < baseline_entry["holdout"]["rmse"]
        ),
        "model_comparison": [
            {
                "model_name": run["model_name"],
                "best_params": run["best_params"],
                "cv_rmse": run["cv_rmse"],
                "holdout": run["holdout"],
            }
            for run in runs
        ],
    }
    mutual_info_report = build_mutual_information_report(best_entry["estimator"], X_train, y_train)
    return best_entry["estimator"], metrics, runs, mutual_info_report


def build_mutual_information_report(
    fitted_model: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> pd.DataFrame:
    """Compute mutual information scores on the encoded training matrix."""
    engineered = fitted_model.named_steps["feature_engineering"].transform(X_train)
    preprocessor = fitted_model.named_steps["preprocessor"]
    transformed = preprocessor.transform(engineered)
    feature_names = get_feature_names_after_preprocessing(preprocessor)
    mi_scores = mutual_info_regression(transformed, y_train, random_state=RANDOM_STATE)
    support_mask = fitted_model.named_steps["variance"].get_support()
    selected_feature_names = np.asarray(feature_names)[support_mask]
    selected_scores = np.asarray(mi_scores)[support_mask]
    report = pd.DataFrame(
        {
            "feature": selected_feature_names,
            "mutual_information": selected_scores,
        }
    )
    return report.sort_values("mutual_information", ascending=False).reset_index(drop=True)


def get_feature_names_after_preprocessing(preprocessor: ColumnTransformer) -> list[str]:
    """Return feature names after numeric scaling and categorical one-hot encoding."""
    numeric_features = list(preprocessor.transformers_[0][2])
    categorical_features = list(preprocessor.transformers_[1][2])
    onehot = preprocessor.named_transformers_["cat"].named_steps["onehot"]
    encoded_categories = list(onehot.get_feature_names_out(categorical_features))
    return numeric_features + encoded_categories


def save_json(path: Path, payload: dict[str, Any]) -> None:
    """Save a JSON artifact with deterministic formatting."""
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def save_artifacts(
    model: Pipeline,
    metrics: dict[str, Any],
    model_info: dict[str, Any],
    input_schema: dict[str, Any],
    input_defaults: dict[str, Any],
    mutual_info_report: pd.DataFrame,
    artifact_dir: Path | None = None,
) -> None:
    """Persist the best model and metadata used by the service."""
    artifact_dir = Path(artifact_dir) if artifact_dir else ARTIFACT_DIR
    artifact_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, artifact_dir / BEST_MODEL_PATH.name)
    save_json(artifact_dir / METRICS_PATH.name, metrics)
    save_json(artifact_dir / MODEL_INFO_PATH.name, model_info)
    save_json(artifact_dir / INPUT_SCHEMA_PATH.name, input_schema)
    save_json(artifact_dir / INPUT_DEFAULTS_PATH.name, input_defaults)
    mutual_info_report.to_csv(artifact_dir / MUTUAL_INFO_PATH.name, index=False)


def load_artifacts(artifact_dir: Path | None = None) -> dict[str, Any]:
    """Load all inference artifacts required by the Flask service."""
    artifact_dir = Path(artifact_dir) if artifact_dir else ARTIFACT_DIR
    required = {
        "model": artifact_dir / BEST_MODEL_PATH.name,
        "metrics": artifact_dir / METRICS_PATH.name,
        "model_info": artifact_dir / MODEL_INFO_PATH.name,
        "input_schema": artifact_dir / INPUT_SCHEMA_PATH.name,
        "input_defaults": artifact_dir / INPUT_DEFAULTS_PATH.name,
    }
    missing = [name for name, path in required.items() if not path.is_file()]
    if missing:
        missing_joined = ", ".join(missing)
        raise FileNotFoundError(
            f"Missing inference artifacts in {artifact_dir}: {missing_joined}. "
            "Run `python exercise_3/main.py train` first."
        )

    return {
        "model": joblib.load(required["model"]),
        "metrics": json.loads(required["metrics"].read_text(encoding="utf-8")),
        "model_info": json.loads(required["model_info"].read_text(encoding="utf-8")),
        "input_schema": json.loads(required["input_schema"].read_text(encoding="utf-8")),
        "input_defaults": json.loads(required["input_defaults"].read_text(encoding="utf-8")),
        "artifact_dir": str(artifact_dir),
    }


def parse_demo_payload(raw_value: str | None) -> list[float]:
    """Parse the `data=[...]` query parameter used by the demo endpoint."""
    if raw_value is None:
        raise ValueError("Query parameter `data` is required.")
    try:
        parsed = ast.literal_eval(raw_value)
    except (SyntaxError, ValueError) as exc:
        raise ValueError("Query parameter `data` must be a list of 5 numbers.") from exc
    if not isinstance(parsed, (list, tuple)) or len(parsed) != len(DEMO_FEATURE_ORDER):
        raise ValueError(
            f"Query parameter `data` must contain exactly {len(DEMO_FEATURE_ORDER)} values."
        )
    values: list[float] = []
    for item in parsed:
        try:
            values.append(float(item))
        except (TypeError, ValueError) as exc:
            raise ValueError("All demo values must be numeric.") from exc
    return values


def build_demo_input_frame(
    values: list[float],
    input_schema: dict[str, Any],
    input_defaults: dict[str, Any],
) -> pd.DataFrame:
    """Create a one-row DataFrame for the demo endpoint."""
    if len(values) != len(DEMO_FEATURE_ORDER):
        raise ValueError(f"Expected {len(DEMO_FEATURE_ORDER)} demo values, got {len(values)}.")
    payload = {feature: value for feature, value in zip(DEMO_FEATURE_ORDER, values)}
    return build_prediction_frame(payload, input_schema, input_defaults)


def build_prediction_frame(
    payload: dict[str, Any],
    input_schema: dict[str, Any],
    input_defaults: dict[str, Any],
) -> pd.DataFrame:
    """Validate and normalize a raw Ames payload for inference."""
    if not isinstance(payload, dict):
        raise ValueError("Prediction payload must be a JSON object.")

    fields = input_schema["fields"]
    unknown = sorted(set(payload) - set(fields))
    if unknown:
        raise ValueError(f"Unknown fields: {', '.join(unknown)}")

    normalized: dict[str, Any] = {}
    for field_name in input_schema["feature_order"]:
        spec = fields[field_name]
        raw_value = payload.get(field_name, input_defaults[field_name])
        if raw_value is None or (isinstance(raw_value, str) and not raw_value.strip()):
            raw_value = input_defaults[field_name]
        normalized[field_name] = normalize_value(field_name, raw_value, spec)
    return pd.DataFrame([normalized], columns=input_schema["feature_order"])


def normalize_value(field_name: str, value: Any, spec: dict[str, Any]) -> Any:
    """Validate a single raw Ames field according to the saved schema."""
    if spec["type"] == "number":
        try:
            return float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Field `{field_name}` must be numeric.") from exc

    if not isinstance(value, str):
        raise ValueError(f"Field `{field_name}` must be a string category.")
    cleaned = value.strip()
    if cleaned not in spec["allowed_values"]:
        allowed_preview = ", ".join(spec["allowed_values"][:10])
        raise ValueError(f"Field `{field_name}` must be one of: {allowed_preview}")
    return cleaned


def predict_price(model: Pipeline, input_frame: pd.DataFrame) -> float:
    """Return a scalar prediction from the trained pipeline."""
    prediction = model.predict(input_frame)
    return float(np.asarray(prediction).ravel()[0])
