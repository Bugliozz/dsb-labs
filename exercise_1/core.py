"""Exercise 1 plotting and Docker helper workflows."""

from __future__ import annotations

import inspect
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import gridspec
from matplotlib.collections import EllipseCollection
from matplotlib.colors import Normalize

BASE_DIR = Path(__file__).resolve().parent
STEP_2_DIR = BASE_DIR / "step_2"
STEP_3_DIR = BASE_DIR / "step_3"
STEP_4_DIR = BASE_DIR / "step_4"
STEP_5_DIR = BASE_DIR / "step_5"
STEP_6_DIR = BASE_DIR / "step_6"
STEP_7_DIR = BASE_DIR / "step_7"


def exercise_1_2() -> None:
    """Generate the exercise 1.2 multi-plot figure and save to disk."""
    np.random.seed(42)
    sns.set_theme(style="darkgrid")

    x_scatter = np.random.rand(100) * 1.8 - 0.8
    y_scatter = 2.0 * x_scatter + np.random.randn(100)

    x_step = np.arange(6)
    y_step = np.array([0, 1, 4, 9, 16, 25])

    x_bar = np.arange(1, 6)
    y_bar = x_bar**2

    x_fill = np.linspace(0, 5, 200)
    y_fill_low = x_fill**2
    y_fill_high = x_fill**3

    dates = pd.date_range(start="2018-01-01", periods=100)
    ts = np.random.randn(100) + np.linspace(-0.8, 1.2, 100)

    box_data = np.random.randn(100)
    hist_data = np.concatenate(
        [
            np.random.normal(0.0, 0.9, 85),
            np.random.normal(2.0, 0.22, 15),
        ]
    )

    fig = plt.figure(figsize=(11, 5.5))
    gs = gridspec.GridSpec(
        2,
        4,
        figure=fig,
        width_ratios=[1, 1, 1, 1],
        height_ratios=[1, 1],
        wspace=0.45,
        hspace=0.55,
    )

    ax_scatter = fig.add_subplot(gs[0, 0])
    ax_step = fig.add_subplot(gs[0, 1])
    ax_bar = fig.add_subplot(gs[0, 2])
    ax_fill = fig.add_subplot(gs[0, 3])
    ax_ts = fig.add_subplot(gs[1, 0:2])
    ax_box = fig.add_subplot(gs[1, 2])
    ax_hist = fig.add_subplot(gs[1, 3])

    for ax in (ax_scatter, ax_step, ax_bar, ax_fill, ax_box, ax_hist):
        ax.set_box_aspect(1)

    ax_scatter.scatter(x_scatter, y_scatter, color="#1f77b4", s=34, edgecolor="k", alpha=0.8)
    ax_scatter.set_title("Scatter Plot")
    ax_scatter.set_xlim(-0.8, 1.1)

    ax_step.step(x_step, y_step, color="#1f77b4", linewidth=2)
    ax_step.set_title("Step Plot")

    ax_bar.bar(x_bar, y_bar, color="#74a9cf")
    ax_bar.set_title("Bar Chart")

    ax_fill.fill_between(x_fill, y_fill_low, y_fill_high, color="#4c78a8", alpha=0.55)
    ax_fill.set_title("Fill Between")

    ax_ts.plot(dates, ts, color="#2f7f9f", linewidth=1.5)
    ax_ts.set_title("Time Series")
    ax_ts.set_xticks([dates[0], dates[30], dates[60], dates[90]])

    ax_box.boxplot(box_data, vert=True, patch_artist=True, boxprops=dict(facecolor="#e6cf87", color="k"))
    ax_box.set_title("Box Plot")

    ax_hist.hist(hist_data, bins=12, color="#2b8cbe", edgecolor="white")
    ax_hist.set_title("Histogram")

    fig.subplots_adjust(left=0.05, right=0.985, top=0.93, bottom=0.1)
    out_path = STEP_2_DIR / "images" / "exercise1_plots.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved figure to {out_path}")
    try:
        plt.show()
    except Exception:
        pass


def exercise_1_3() -> None:
    """Generate the multi-chart figure from the exercise description."""
    sns.set_theme(style="darkgrid")

    years = np.arange(2000, 2006)
    apples = [0.35, 0.6, 0.9, 0.8, 0.65, 0.8]
    oranges = [0.4, 0.8, 0.9, 0.7, 0.6, 0.8]

    fig, axs = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)

    ax = axs[0, 0]
    ax.plot(years, apples, marker="s", color="blue", label="Apples")
    ax.plot(years, oranges, marker="o", linestyle="--", color="red", label="Oranges")
    ax.set_title("Crop Yields in Kanto")
    ax.set_xlabel("Year")
    ax.set_ylabel("Yield (tons per hectare)")
    ax.legend(loc="lower right")

    iris = sns.load_dataset("iris")
    sns.scatterplot(
        data=iris,
        x="sepal_length",
        y="sepal_width",
        hue="species",
        palette=["#4c72b0", "#dd8452", "#55a868"],
        ax=axs[0, 1],
    )
    axs[0, 1].set_title("Sepal Length vs. Sepal Width")

    tips = sns.load_dataset("tips")
    barplot_kwargs = {
        "data": tips,
        "x": "day",
        "y": "total_bill",
        "hue": "sex",
        "palette": "deep",
        "ax": axs[1, 0],
    }
    if "errorbar" in inspect.signature(sns.barplot).parameters:
        barplot_kwargs["errorbar"] = "sd"
    else:
        barplot_kwargs["ci"] = "sd"
    sns.barplot(**barplot_kwargs)
    axs[1, 0].set_title("Restaurant bills")
    axs[1, 0].legend(title="sex", loc="lower left")

    flights = sns.load_dataset("flights")
    flights_pivot = flights.pivot(index="month", columns="year", values="passengers")
    sns.heatmap(flights_pivot, cmap="Blues", cbar_kws={"label": "passengers"}, ax=axs[1, 1])
    axs[1, 1].set_title("Flight traffic")
    axs[1, 1].set_xlabel("year")
    axs[1, 1].set_ylabel("month")

    out_path = STEP_3_DIR / "images" / "exercise1_3.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    print(f"Saved figure to {out_path}")
    try:
        plt.show()
    except Exception:
        pass


def _find_liver_dataset() -> Path | None:
    """Return the first existing liver CSV path from common names/locations."""
    dataset_dir = STEP_4_DIR / "datasets"
    common_names = [
        dataset_dir / "indian_liver_patient.csv",
        dataset_dir / "Indian Liver Patient Dataset (ILPD).csv",
        dataset_dir / "indian_liver_patient_records.csv",
        dataset_dir / "ILPD.csv",
    ]
    for candidate in common_names:
        if candidate.is_file():
            return candidate

    recursive_patterns = ["**/*liver*.csv", "**/*Liver*.csv", "**/*ilpd*.csv", "**/*ILPD*.csv"]
    for pattern in recursive_patterns:
        for candidate in dataset_dir.glob(pattern):
            if candidate.is_file():
                return candidate
    return None


def _prepare_liver_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Clean column names and align fields with the class exercise."""
    df = df.copy()
    df.columns = [col.strip().replace(" ", "_") for col in df.columns]
    df = df.rename(
        columns={
            "A/G_Ratio": "Albumin_and_Globulin_Ratio",
            "Albumin_and_Globulin_Ratio__A/G_Ratio": "Albumin_and_Globulin_Ratio",
            "Dataset": "Target",
        }
    )

    if "Gender" in df.columns:
        gender_norm = df["Gender"].astype(str).str.strip().str.lower()
        df["Gender"] = gender_norm.map({"male": True, "female": False})

    if "Albumin_and_Globulin_Ratio" in df.columns:
        df["Albumin_and_Globulin_Ratio"] = df["Albumin_and_Globulin_Ratio"].fillna(
            df["Albumin_and_Globulin_Ratio"].median()
        )

    return df


def exercise_1_4() -> None:
    """Perform exploratory analysis on the Indian Liver dataset."""
    data_path = _find_liver_dataset()
    if data_path is None:
        print("[Exercise 1.4] dataset not found.")
        print(f"Expected inside: {STEP_4_DIR / 'datasets'}")
        return

    print(f"[Exercise 1.4] using dataset: {data_path}")
    df = pd.read_csv(data_path)
    df = _prepare_liver_dataframe(df)
    image_dir = STEP_4_DIR / "images"
    image_dir.mkdir(parents=True, exist_ok=True)

    required_cols = {
        "Target",
        "Gender",
        "Age",
        "Total_Bilirubin",
        "Direct_Bilirubin",
        "Alkaline_Phosphotase",
        "Alamine_Aminotransferase",
        "Aspartate_Aminotransferase",
        "Albumin",
    }
    missing = sorted(required_cols - set(df.columns))
    if missing:
        print(f"[Exercise 1.4] missing required columns: {missing}")
        print(f"Available columns: {list(df.columns)}")
        return

    print("\n[Exercise 1.4] Data overview:")
    print(f"There are {len(df)} rows and {len(df.columns)} columns")
    print("\nColumn dtypes:")
    print(df.dtypes)
    print("\nValue counts for target (Target):")
    print(df["Target"].value_counts().sort_index())

    corr = df.corr(numeric_only=True)
    fig_corr, ax_corr = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax_corr)
    ax_corr.set_title("Feature correlations")
    corr_path = image_dir / "exercise1_4_correlation.png"
    fig_corr.savefig(corr_path, dpi=150, bbox_inches="tight")
    print(f"Saved correlation heatmap to {corr_path}")
    plt.close(fig_corr)

    numeric_cols = [
        "Age",
        "Total_Bilirubin",
        "Direct_Bilirubin",
        "Alkaline_Phosphotase",
        "Alamine_Aminotransferase",
        "Aspartate_Aminotransferase",
        "Total_Protiens",
        "Albumin",
        "Albumin_and_Globulin_Ratio",
    ]
    numeric_cols = [col for col in numeric_cols if col in df.columns]
    pair_grid = sns.pairplot(
        df[numeric_cols + ["Target"]],
        hue="Target",
        corner=True,
        plot_kws={"alpha": 0.5, "s": 20},
    )
    pair_path = image_dir / "exercise1_4_pairplot.png"
    pair_grid.fig.savefig(pair_path, dpi=150, bbox_inches="tight")
    print(f"Saved pairplot to {pair_path}")
    plt.close(pair_grid.fig)

    fig_target, ax_target = plt.subplots(figsize=(6, 4))
    sns.countplot(data=df, x="Target", palette="Blues", ax=ax_target)
    ax_target.set_title("Target distribution (1=liver disease, 2=healthy)")
    target_path = image_dir / "exercise1_4_target_distribution.png"
    fig_target.savefig(target_path, dpi=150, bbox_inches="tight")
    print(f"Saved target distribution plot to {target_path}")
    plt.close(fig_target)

    fig_box, ax_box = plt.subplots(figsize=(12, 6))
    measures = [
        "Total_Bilirubin",
        "Direct_Bilirubin",
        "Alkaline_Phosphotase",
        "Alamine_Aminotransferase",
        "Aspartate_Aminotransferase",
        "Total_Protiens",
        "Albumin",
        "Albumin_and_Globulin_Ratio",
    ]
    measures = [col for col in measures if col in df.columns]
    melted = df.melt(
        id_vars=["Target"],
        value_vars=measures,
        var_name="measure",
        value_name="value",
    )
    sns.boxplot(data=melted, x="measure", y="value", hue="Target", palette="Set2", ax=ax_box)
    ax_box.tick_params(axis="x", rotation=45)
    ax_box.set_title("Measurements by class")
    box_path = image_dir / "exercise1_4_boxplots.png"
    fig_box.savefig(box_path, dpi=150, bbox_inches="tight")
    print(f"Saved measurement boxplots to {box_path}")
    plt.close(fig_box)

    if "Albumin_and_Globulin_Ratio" in df.columns:
        fig_agr, ax_agr = plt.subplots(figsize=(8, 5))
        sns.boxplot(data=df, x="Target", y="Albumin_and_Globulin_Ratio", ax=ax_agr)
        ax_agr.set_title("Albumin_and_Globulin_Ratio by Target")
        agr_path = image_dir / "exercise1_4_ag_ratio_boxplot.png"
        fig_agr.savefig(agr_path, dpi=150, bbox_inches="tight")
        print(f"Saved A/G ratio boxplot to {agr_path}")
        plt.close(fig_agr)

    if "Direct_Bilirubin" in df.columns:
        fig_dir, ax_dir = plt.subplots(figsize=(8, 5))
        sns.swarmplot(data=df, x="Target", y="Direct_Bilirubin", size=3, ax=ax_dir)
        ax_dir.set_title("Direct_Bilirubin by Target")
        dir_path = image_dir / "exercise1_4_direct_bilirubin_swarm.png"
        fig_dir.savefig(dir_path, dpi=150, bbox_inches="tight")
        print(f"Saved Direct_Bilirubin swarm plot to {dir_path}")
        plt.close(fig_dir)

    summary_cols = [
        "Total_Bilirubin",
        "Direct_Bilirubin",
        "Alamine_Aminotransferase",
        "Aspartate_Aminotransferase",
        "Albumin",
        "Albumin_and_Globulin_Ratio",
    ]
    summary_cols = [col for col in summary_cols if col in df.columns]
    target_means = df.groupby("Target")[summary_cols].mean().round(3)
    print("\nMean values by target:")
    print(target_means)
    print("\nConclusions (EDA):")
    print("- The two target classes show only partially separated distributions.")
    print("- No single feature clearly separates healthy and liver-disease classes.")
    print("- Bilirubin and aminotransferase variables are strongly informative together.")
    print("- A multivariate ML model is needed for better separation.")


def _plot_corr_ellipses(data: pd.DataFrame | np.ndarray, figsize=None, **kwargs):
    """Plot a correlation matrix using oriented ellipses."""
    matrix = np.array(data)
    if matrix.ndim != 2:
        raise ValueError("data must be a 2D array")

    fig, ax = plt.subplots(1, 1, figsize=figsize, subplot_kw={"aspect": "equal"})
    ax.set_xlim(-0.5, matrix.shape[1] - 0.5)
    ax.set_ylim(-0.5, matrix.shape[0] - 0.5)
    ax.invert_yaxis()

    xy = np.indices(matrix.shape)[::-1].reshape(2, -1).T
    widths = np.ones_like(matrix).ravel() + 0.01
    heights = 1 - np.abs(matrix).ravel() - 0.01
    angles = 45 * np.sign(matrix).ravel()

    ellipses = EllipseCollection(
        widths=widths,
        heights=heights,
        angles=angles,
        units="x",
        offsets=xy,
        norm=Normalize(vmin=-1, vmax=1),
        transOffset=ax.transData,
        array=matrix.ravel(),
        **kwargs,
    )
    ax.add_collection(ellipses)

    if isinstance(data, pd.DataFrame):
        ax.set_xticks(np.arange(matrix.shape[1]))
        ax.set_xticklabels(data.columns, rotation=90)
        ax.set_yticks(np.arange(matrix.shape[0]))
        ax.set_yticklabels(data.index)

    return ellipses, ax


def exercise_1_5() -> None:
    """Replicate ETF correlation plots from PS4DS Chapter 1 notebook."""
    sectors_path = STEP_5_DIR / "datasets" / "sp500_sectors.csv"
    prices_path = STEP_5_DIR / "datasets" / "sp500_data.csv.gz"
    image_dir = STEP_5_DIR / "images"
    image_dir.mkdir(parents=True, exist_ok=True)

    if not sectors_path.is_file() or not prices_path.is_file():
        print("[Exercise 1.5] dataset files not found.")
        print(f"Expected: {sectors_path} and {prices_path}")
        return

    sp500_sym = pd.read_csv(sectors_path)
    sp500_px = pd.read_csv(prices_path, index_col=0)
    sp500_px.index = pd.to_datetime(sp500_px.index)

    etf_symbols = sp500_sym.loc[sp500_sym["sector"] == "etf", "symbol"].tolist()
    etf_symbols = [sym for sym in etf_symbols if sym in sp500_px.columns]
    if not etf_symbols:
        print("[Exercise 1.5] no ETF symbols found in the price table.")
        return

    etfs = sp500_px.loc[sp500_px.index > "2012-07-01", etf_symbols]
    etfs = etfs.dropna(axis=0, how="all").dropna(axis=1, how="all")
    corr = etfs.corr(numeric_only=True)

    fig_heat, ax_heat = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        corr,
        vmin=-1,
        vmax=1,
        cmap=sns.diverging_palette(20, 220, as_cmap=True),
        ax=ax_heat,
    )
    heatmap_path = image_dir / "exercise1_5_etf_correlation_heatmap.png"
    fig_heat.savefig(heatmap_path, dpi=150, bbox_inches="tight")
    print(f"Saved ETF correlation heatmap to {heatmap_path}")
    plt.close(fig_heat)

    mappable, ax_ell = _plot_corr_ellipses(corr, figsize=(5, 4), cmap="bwr_r")
    colorbar = plt.colorbar(mappable, ax=ax_ell)
    colorbar.set_label("Correlation coefficient")
    ellipses_path = image_dir / "exercise1_5_etf_correlation_ellipses.png"
    plt.tight_layout()
    plt.savefig(ellipses_path, dpi=150, bbox_inches="tight")
    print(f"Saved ETF correlation ellipses to {ellipses_path}")
    try:
        plt.show()
    except Exception:
        pass
    plt.close()


def _report_required_assets(exercise_id: str, step_dir: Path, files: list[str], commands: list[str]) -> None:
    """Validate a scaffold-only exercise and print the next steps."""
    missing = [name for name in files if not (step_dir / name).exists()]
    if missing:
        print(f"[Exercise {exercise_id}] missing committed assets: {', '.join(missing)}")
        print(f"[Exercise {exercise_id}] expected folder: {step_dir}")
        return

    print(f"[Exercise {exercise_id}] committed assets verified in {step_dir}")
    if exercise_id == "1_6" and not (step_dir / "dokuwiki_data").exists():
        print("[Exercise 1.6] note: 'dokuwiki_data' will be created on first Docker run if missing.")
    print(f"[Exercise {exercise_id}] next steps:")
    for command in commands:
        print(f"  - {command}")
    print(f"[Exercise {exercise_id}] see README: {step_dir / 'README.md'}")


def exercise_1_6() -> None:
    """Validate the Docker/DokuWiki scaffold without mutating tracked files."""
    _report_required_assets(
        "1_6",
        STEP_6_DIR,
        ["docker-compose.yml", "README.md"],
        [
            "cd exercise_1/step_6",
            "docker compose up -d",
            "docker compose down",
        ],
    )


def exercise_1_7() -> None:
    """Validate the Flask + Docker scaffold without rewriting local assets."""
    _report_required_assets(
        "1_7",
        STEP_7_DIR,
        ["app.py", "requirements.txt", "Dockerfile", "README.md"],
        [
            "python -m pip install -r exercise_1/requirements.txt",
            "python exercise_1/step_7/app.py",
            "docker build --tag python-docker exercise_1/step_7",
        ],
    )


EXERCISE_HANDLERS = {
    "1_2": exercise_1_2,
    "1_3": exercise_1_3,
    "1_4": exercise_1_4,
    "1_5": exercise_1_5,
    "1_6": exercise_1_6,
    "1_7": exercise_1_7,
}


def run_exercises(exercise_ids: list[str]) -> None:
    """Run a list of Exercise 1 steps in order."""
    for exercise_id in exercise_ids:
        print(f"\n[Runner] running exercise {exercise_id}")
        EXERCISE_HANDLERS[exercise_id]()
