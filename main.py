"""Main script for plotting exercises.

Contains functions corresponding to the class assignments:

* `exercise_1_2()` recreates the multi-panel figure from exercise 1.2
* `exercise_1_3()` builds the composed Seaborn figure shown in the problem
* `exercise_1_4()` optional exploratory analysis on the Indian liver data
* `exercise_1_5()` replicates the ETF correlation plots from PS4DS Chapter 1
* `exercise_1_6()` prepares a Docker/DokuWiki lab scaffold
* `exercise_1_7()` prepares a Flask + Docker lab scaffold
* `exercise_2_1()` runs the Kaggle + MySQL lab workflow
* `exercise_2_2()` guides the Metabase + MySQL exploration workflow
* `exercise_2_3()` guides the Neo4j Browser tutorials + Python Cypher workflow
* `exercise_2_4()` guides the OpenSearch dashboard exploration workflow

The functions are called from the standard ``if __name__ == '__main__'``
guard at the bottom; importing this module does not execute any plotting by
itself.
"""

import inspect
import argparse
import os
import shutil
import urllib.request
import urllib.error
from base64 import b64encode
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import gridspec
from matplotlib.collections import EllipseCollection
from matplotlib.colors import Normalize


def _read_csv_with_auto_separator(csv_path: Path) -> tuple[pd.DataFrame, str]:
    """Read a CSV trying comma first and semicolon as fallback."""
    df = pd.read_csv(csv_path)
    if len(df.columns) == 1:
        df = pd.read_csv(csv_path, sep=";")
        return df, ";"
    return df, ","


def _convert_text_binary_columns(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Convert binary text columns (yes/no etc.) to integer 0/1."""
    converted = {}
    out = df.copy()
    for col in out.columns:
        if not (pd.api.types.is_object_dtype(out[col]) or pd.api.types.is_string_dtype(out[col])):
            continue

        normalized = out[col].dropna().astype(str).str.strip().str.lower()
        unique_vals = set(normalized.unique())
        mapping = None
        if unique_vals == {"yes", "no"}:
            mapping = {"no": 0, "yes": 1}
        elif unique_vals == {"true", "false"}:
            mapping = {"false": 0, "true": 1}
        elif unique_vals == {"y", "n"}:
            mapping = {"n": 0, "y": 1}

        if mapping is None:
            continue

        out[col] = out[col].astype(str).str.strip().str.lower().map(mapping).astype("Int64")
        converted[col] = mapping

    return out, converted


def _download_dataset_csvs(dataset_id: str, target_dir: Path, kagglehub_module) -> list[Path]:
    """Download a Kaggle dataset and copy all CSV files to target_dir."""
    cache_path = Path(kagglehub_module.dataset_download(dataset_id))
    csv_files = sorted(cache_path.rglob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found for dataset '{dataset_id}'")

    target_dir.mkdir(parents=True, exist_ok=True)
    copied_files = []
    for src in csv_files:
        dest = target_dir / src.name
        shutil.copy2(src, dest)
        copied_files.append(dest)
    return copied_files


def _existing_csvs(target_dir: Path) -> list[Path]:
    """Return existing CSV files in target_dir."""
    if not target_dir.is_dir():
        return []
    return sorted(target_dir.rglob("*.csv"))


def _ensure_dataset_csvs(
    dataset_id: str,
    target_dir: Path,
    kagglehub_module,
    force_download: bool = False,
) -> list[Path]:
    """Reuse local CSVs when available, otherwise download from Kaggle."""
    existing_files = _existing_csvs(target_dir)
    if existing_files and not force_download:
        return existing_files
    return _download_dataset_csvs(dataset_id, target_dir, kagglehub_module)


def _pick_bank_marketing_csv(csv_files: list[Path]) -> Path:
    """Pick the most likely Bank Marketing CSV from a list of files."""
    for csv_file in csv_files:
        name = csv_file.stem.lower()
        if "bank" in name and "marketing" in name:
            return csv_file
    for csv_file in csv_files:
        if "bank" in csv_file.stem.lower():
            return csv_file
    return csv_files[0]


def exercise_1_2():
    """Generate the exercise 1.2 multi-plot figure and save to disk."""

    # Ensure reproducibility and style close to the slide screenshot.
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

    # Keep panel proportions consistent across different backends/displays.
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
    out_path = Path("exercise_1_2/images/exercise1_plots.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved figure to {out_path}")
    try:
        plt.show()
    except Exception:
        pass


# --- Exercise 1.3: replicate composed Seaborn plots ---
def exercise_1_3():
    """Generate the multi-chart figure from the exercise description.

    Layout is a 2x2 grid containing:
    1. Crop yields line chart (apples/oranges)
    2. Iris sepal-length vs sepal-width scatterplot
    3. Tips dataset barplot of total bills by day/sex
    4. Flights dataset heatmap of passengers by month/year
    """
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

    out_path = Path("exercise_1_3/images/exercise1_3.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    print(f"Saved figure to {out_path}")
    try:
        plt.show()
    except Exception:
        pass


# --- Exercise 1.4: Indian Liver Patient Records EDA ---
def _find_liver_dataset():
    """Return the first existing liver CSV path from common names/locations."""
    common_names = [
        "exercise_1_4/datasets/indian_liver_patient.csv",
        "indian_liver_patient.csv",
        "Indian Liver Patient Dataset (ILPD).csv",
        "indian_liver_patient_records.csv",
        "ILPD.csv",
    ]
    search_roots = [Path("."), Path("..")]

    for root in search_roots:
        for name in common_names:
            candidate = root / name
            if candidate.is_file():
                return candidate

    for root in search_roots:
        for candidate in root.glob("*liver*.csv"):
            if candidate.is_file():
                return candidate

    # Fallback: also search recursively in likely folders and ILPD naming variants.
    recursive_patterns = ["**/*liver*.csv", "**/*Liver*.csv", "**/*ilpd*.csv", "**/*ILPD*.csv"]
    for root in search_roots:
        for pattern in recursive_patterns:
            for candidate in root.glob(pattern):
                if candidate.is_file() and ".venv" not in str(candidate):
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
        # Boolean encoding requested in the exercise material.
        df["Gender"] = gender_norm.map({"male": True, "female": False})

    if "Albumin_and_Globulin_Ratio" in df.columns:
        df["Albumin_and_Globulin_Ratio"] = df["Albumin_and_Globulin_Ratio"].fillna(
            df["Albumin_and_Globulin_Ratio"].median()
        )

    return df


def exercise_1_4():
    """Perform exploratory analysis and visualization on the
    "Indian Liver Patient Records" dataset (optional exercise).

    The dataset is **not** bundled with the repository; download it from
    Kaggle and place it in ``exercise_1_4/datasets/`` with the name
    ``indian_liver_patient.csv``. See README for details.
    """
    data_path = _find_liver_dataset()
    if data_path is None:
        print("[Exercise 1.4] dataset not found.")
        print("Expected one of: indian_liver_patient.csv, Indian Liver Patient Dataset (ILPD).csv")
        print(f"Current working directory: {Path.cwd()}")
        print("Place the CSV in exercise_1_4/datasets/ (or another subfolder) and rerun.")
        return

    print(f"[Exercise 1.4] using dataset: {data_path}")
    df = pd.read_csv(data_path)
    df = _prepare_liver_dataframe(df)
    image_dir = Path("exercise_1_4/images")
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
    cat2 = image_dir / "exercise1_4_boxplots.png"
    fig_box.savefig(cat2, dpi=150, bbox_inches="tight")
    print(f"Saved measurement boxplots to {cat2}")
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
    target_means = (
        df.groupby("Target")[summary_cols]
        .mean()
        .round(3)
    )
    print("\nMean values by target:")
    print(target_means)
    print("\nConclusions (EDA):")
    print("- The two target classes show only partially separated distributions.")
    print("- No single feature clearly separates healthy and liver-disease classes.")
    print("- Bilirubin and aminotransferase variables are strongly informative together.")
    print("- A multivariate ML model is needed for better separation.")


def _plot_corr_ellipses(data, figsize=None, **kwargs):
    """Plot a correlation matrix using oriented ellipses.

    Adapted from https://stackoverflow.com/a/34558488, as used in the
    Practical Statistics for Data Scientists notebook.
    """
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


def exercise_1_5():
    """Replicate ETF correlation plots from PS4DS Chapter 1 notebook.

    Source notebook:
    https://github.com/gedeck/practical-statistics-for-data-scientists/
    """
    sectors_path = Path("exercise_1_5/datasets/sp500_sectors.csv")
    prices_path = Path("exercise_1_5/datasets/sp500_data.csv.gz")
    image_dir = Path("exercise_1_5/images")
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


def exercise_1_6():
    """Prepare the Docker/DokuWiki exercise scaffold and usage instructions."""
    exercise_dir = Path("exercise_1_6")
    data_dir = exercise_dir / "dokuwiki_data"
    compose_path = exercise_dir / "docker-compose.yml"
    guide_path = exercise_dir / "README.md"

    exercise_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    compose_content = """services:
  dokuwiki:
    image: dokuwiki/dokuwiki:stable
    container_name: dokuwiki-lab
    ports:
      - "8080:8080"
    user: "1000:1000"
    volumes:
      - ./dokuwiki_data:/storage
    restart: unless-stopped
"""

    guide_content = """# Exercise 1.6 - Docker DokuWiki

This folder replicates the Docker + DokuWiki workflow from class.

## 1. Start Docker

- Linux (systemd):
  `sudo systemctl start docker.service`
- Windows/macOS:
  start Docker Desktop and wait for "Engine running".

## 2. Start the container

From this folder (`exercise_1_6/`):

```bash
docker compose up -d
```

Open:
`http://localhost:8080/`

## 3. Check process and logs

```bash
docker ps
docker logs dokuwiki-lab
docker exec -it dokuwiki-lab bash
```

## 4. Check persistent files

```bash
ls dokuwiki_data
```

On Windows PowerShell:

```powershell
Get-ChildItem .\\dokuwiki_data
```

## 5. Stop and remove the container

```bash
docker compose down
```

## 6. List local Docker images

```bash
docker image ls
```

Equivalent one-shot command (as shown in slides):

```bash
docker run -d -p 8080:8080 --user 1000:1000 -v ./dokuwiki_data:/storage --name dokuwiki-lab dokuwiki/dokuwiki:stable
```
"""

    compose_path.write_text(compose_content, encoding="utf-8")
    guide_path.write_text(guide_content, encoding="utf-8")

    print(f"Created: {compose_path}")
    print(f"Created: {guide_path}")
    print(f"Persistent data directory: {data_dir}")
    print("Run `docker compose up -d` inside exercise_1_6 to start DokuWiki.")


def exercise_1_7():
    """Prepare the Flask + Docker exercise scaffold and usage instructions."""
    exercise_dir = Path("exercise_1_7")
    app_path = exercise_dir / "app.py"
    requirements_path = exercise_dir / "requirements.txt"
    dockerfile_path = exercise_dir / "Dockerfile"
    guide_path = exercise_dir / "README.md"

    exercise_dir.mkdir(parents=True, exist_ok=True)

    app_content = """from flask import Flask

app = Flask(__name__)


@app.route("/")
def hello():
    return "<h1>Hello from Flask & Docker</h1>"


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
"""

    requirements_content = """Flask
"""

    dockerfile_content = """FROM python:3.8-slim-buster
WORKDIR /python-docker
COPY requirements.txt app.py ./
RUN pip install --upgrade pip
RUN pip install -r requirements.txt
CMD ["python3", "app.py"]
"""

    guide_content = """# Exercise 1.7 - Flask + Docker

This folder replicates the Flask + Docker workflow from class.

## 1. Run locally (without Docker)

```bash
pip install -r requirements.txt
python app.py
```

Open:
`http://localhost:5000/`

## 2. Build Docker image

```bash
docker build --tag python-docker .
```

## 3. Run container

```bash
docker run -d -p 5000:5000 --name python-docker-lab python-docker
docker ps
```

Open:
`http://localhost:5000/`

## 4. Stop and remove container

```bash
docker stop python-docker-lab
docker rm python-docker-lab
```

## 5. Optional cleanup

```bash
docker image rm python-docker
```
"""

    app_path.write_text(app_content, encoding="utf-8")
    requirements_path.write_text(requirements_content, encoding="utf-8")
    dockerfile_path.write_text(dockerfile_content, encoding="utf-8")
    guide_path.write_text(guide_content, encoding="utf-8")

    print(f"Created: {app_path}")
    print(f"Created: {requirements_path}")
    print(f"Created: {dockerfile_path}")
    print(f"Created: {guide_path}")
    print("Run `python app.py` or build the Docker image inside exercise_1_7.")


def exercise_2_1():
    """Run the full Kaggle + MySQL workflow for exercise 2.1."""
    exercise_dir = Path("exercise_2_1")
    compose_path = exercise_dir / "compose.yaml"
    dataset_dir = exercise_dir / "datasets"
    living_wage_dir = dataset_dir / "living_wage_50_states"
    bank_marketing_dir = dataset_dir / "bank_marketing"
    dataset_dir.mkdir(parents=True, exist_ok=True)

    if compose_path.is_file():
        print(f"[Exercise 2.1] using compose file: {compose_path}")
        print("[Exercise 2.1] start services with: docker compose -f exercise_2_1/compose.yaml up -d")
    else:
        print(f"[Exercise 2.1] warning: compose file not found at {compose_path}")

    try:
        import kagglehub
    except ImportError:
        print("[Exercise 2.1] kagglehub is not installed.")
        print("Install it with: pip install kagglehub")
        return

    force_download = os.getenv("FORCE_KAGGLE_DOWNLOAD", "0").strip().lower() in {"1", "true", "yes", "y"}

    living_wage_id = "brandonconrady/living-wage-50-states"
    print(f"[Exercise 2.1] downloading dataset: {living_wage_id}")
    try:
        living_wage_csvs = _ensure_dataset_csvs(
            living_wage_id,
            living_wage_dir,
            kagglehub,
            force_download=force_download,
        )
        print(f"[Exercise 2.1] {len(living_wage_csvs)} CSV file(s) ready in {living_wage_dir}")
    except Exception as exc:
        living_wage_csvs = []
        print(f"[Exercise 2.1] living wage download failed: {exc}")
        print("Make sure your Kaggle credentials are configured correctly.")

    bank_dataset_candidates = [
        "fedesoriano/bank-marketing",
        "janiobachmann/bank-marketing-dataset",
        "rouseguy/bankbalanced",
    ]
    custom_bank_dataset = os.getenv("BANK_MARKETING_DATASET", "").strip()
    if custom_bank_dataset:
        print(f"[Exercise 2.1] custom Bank Marketing dataset from env: {custom_bank_dataset}")
        bank_dataset_candidates.insert(0, custom_bank_dataset)

    bank_csvs = _existing_csvs(bank_marketing_dir) if not force_download else []
    used_bank_dataset = "local cache" if bank_csvs else None
    if not bank_csvs:
        for dataset_id in bank_dataset_candidates:
            print(f"[Exercise 2.1] trying Bank Marketing dataset: {dataset_id}")
            try:
                bank_csvs = _download_dataset_csvs(dataset_id, bank_marketing_dir, kagglehub)
                used_bank_dataset = dataset_id
                break
            except Exception as exc:
                print(f"[Exercise 2.1] failed for {dataset_id}: {exc}")

    if not bank_csvs:
        print("[Exercise 2.1] unable to download a Bank Marketing dataset.")
        print("Set BANK_MARKETING_DATASET with a valid Kaggle dataset ID and run again.")
        return

    print(f"[Exercise 2.1] using Bank Marketing dataset: {used_bank_dataset}")
    bank_csv_path = _pick_bank_marketing_csv(bank_csvs)
    bank_df, sep_used = _read_csv_with_auto_separator(bank_csv_path)
    print(
        f"[Exercise 2.1] loaded {bank_csv_path.name} ({len(bank_df)} rows x {len(bank_df.columns)} columns, sep='{sep_used}')"
    )

    converted_df, converted_map = _convert_text_binary_columns(bank_df)
    if converted_map:
        print("[Exercise 2.1] converted binary text columns:")
        for col, mapping in converted_map.items():
            print(f"  - {col}: {mapping}")
    else:
        print("[Exercise 2.1] no text binary columns converted.")

    processed_csv = bank_marketing_dir / "bank_marketing_processed.csv"
    converted_df.to_csv(processed_csv, index=False)
    print(f"[Exercise 2.1] saved processed CSV: {processed_csv}")

    try:
        from sqlalchemy import create_engine, text
    except ImportError:
        print("[Exercise 2.1] SQLAlchemy is not installed.")
        print("Install with: pip install sqlalchemy mysql-connector-python")
        return

    db_user = os.getenv("MYSQL_USER", "root").strip() or "root"
    db_password = os.getenv("MYSQL_PASSWORD", "pass")
    db_host = os.getenv("MYSQL_HOST", "localhost").strip() or "localhost"
    db_port = int(os.getenv("MYSQL_PORT", "3306").strip() or "3306")
    db_name = os.getenv("MYSQL_DATABASE", "test").strip() or "test"

    print(
        f"[Exercise 2.1] MySQL target: host={db_host} port={db_port} db={db_name} user={db_user}"
    )

    server_url = f"mysql+mysqlconnector://{db_user}:{db_password}@{db_host}:{db_port}"
    db_url = f"{server_url}/{db_name}"
    try:
        server_engine = create_engine(server_url, echo=False)
        with server_engine.connect() as conn:
            conn.execute(text(f"CREATE DATABASE IF NOT EXISTS `{db_name}`"))
            conn.commit()
        server_engine.dispose()

        engine = create_engine(db_url, echo=False)
        converted_df.to_sql(name="bankmarketing", con=engine, if_exists="replace", index=False)
        preview = pd.read_sql("SELECT * FROM bankmarketing LIMIT 5", engine)
        print("\n[Exercise 2.1] MySQL table 'bankmarketing' preview:")
        print(preview.to_string(index=False))

        converted_df.to_sql(name="bankmarketing_copy", con=engine, if_exists="replace", index=False)
        print("[Exercise 2.1] created table copy: bankmarketing_copy")

        if living_wage_csvs:
            living_wage_path = living_wage_csvs[0]
            living_wage_df, _ = _read_csv_with_auto_separator(living_wage_path)
            living_wage_df.to_sql(name="livingwage50states", con=engine, if_exists="replace", index=False)
            living_preview = pd.read_sql("SELECT * FROM livingwage50states LIMIT 5", engine)
            print("\n[Exercise 2.1] MySQL table 'livingwage50states' preview:")
            print(living_preview.to_string(index=False))
            living_wage_df.to_sql(name="livingwage50states_copy", con=engine, if_exists="replace", index=False)
            print("[Exercise 2.1] created table copy: livingwage50states_copy")

        engine.dispose()
    except Exception as exc:
        print(f"[Exercise 2.1] MySQL upload failed: {exc}")
        print("Verify docker services are running and phpMyAdmin can access db/root/pass.")


def exercise_2_2():
    """Guide and validate the Metabase + MySQL workflow for exercise 2.2."""
    print("[Exercise 2.2] Metabase URL: http://localhost:3000")
    print("[Exercise 2.2] target MySQL connection:")
    print("  host=db")
    print("  port=3306")
    print("  database=test")
    print("  username=root")
    print("  password=pass")

    compose_path = Path("exercise_2_1/compose.yaml")
    if compose_path.is_file():
        print("\n[Exercise 2.2] if services are not running, start them with:")
        print("  docker compose -f exercise_2_1/compose.yaml up -d")
    else:
        print("\n[Exercise 2.2] warning: compose file not found at exercise_2_1/compose.yaml")

    try:
        from sqlalchemy import create_engine, text
    except ImportError:
        print("\n[Exercise 2.2] SQLAlchemy not found.")
        print("Install with: pip install sqlalchemy mysql-connector-python")
        return

    # `localhost` is used from the host machine; Metabase container should use host `db`.
    host_db_url = "mysql+mysqlconnector://root:pass@localhost:3306/test"
    try:
        engine = create_engine(host_db_url, echo=False)
        with engine.connect() as conn:
            tables = conn.execute(text("SHOW TABLES")).fetchall()
        engine.dispose()
    except Exception as exc:
        print(f"\n[Exercise 2.2] MySQL host-side check failed: {exc}")
        print("Make sure Docker services are up and MySQL is reachable on localhost:3306.")
        return

    table_names = [row[0] for row in tables]
    print(f"\n[Exercise 2.2] MySQL reachable from host. Tables found: {len(table_names)}")
    if table_names:
        for name in table_names[:10]:
            print(f"  - {name}")
        if len(table_names) > 10:
            print(f"  ... and {len(table_names) - 10} more")
    else:
        print("No tables found. Run: python main.py --exercise 2_1")

    print("\n[Exercise 2.2] Metabase setup checklist:")
    print("1. Open http://localhost:3000 and complete initial admin login.")
    print("2. Add database -> MySQL.")
    print("3. Use host=db, port=3306, db=test, username=root, password=pass.")
    print("4. Save and wait for sync.")
    print("5. Explore data from 'Browse data' and build simple questions/charts.")


def exercise_2_3():
    """Guide and validate the Neo4j + Python workflow for exercise 2.3."""
    print("[Exercise 2.3] Neo4j Browser URL: http://localhost:7474")
    print("[Exercise 2.3] credentials:")
    print("  username=neo4j")
    print("  password=test12345")
    print("\n[Exercise 2.3] complete these Browser tutorials:")
    print("1. Getting started with Neo4j Browser")
    print("2. Try Neo4j with live data")
    print("3. Cypher basics")

    compose_path = Path("exercise_2_1/compose.yaml")
    if compose_path.is_file():
        print("\n[Exercise 2.3] if services are not running, start them with:")
        print("  docker compose -f exercise_2_1/compose.yaml up -d")

    try:
        from neo4j import GraphDatabase
    except ImportError:
        print("\n[Exercise 2.3] neo4j driver not found.")
        print("Install with: pip install neo4j")
        return

    uri = os.getenv("NEO4J_URI", "neo4j://localhost:7687").strip() or "neo4j://localhost:7687"
    user = os.getenv("NEO4J_USER", "neo4j").strip() or "neo4j"
    password = os.getenv("NEO4J_PASSWORD", "test12345")

    print(f"\n[Exercise 2.3] testing Python driver connection: uri={uri} user={user}")

    try:
        with GraphDatabase.driver(uri, auth=(user, password)) as driver:
            driver.verify_connectivity()
            print("[Exercise 2.3] connectivity check: OK")

            queries = [
                (
                    "People sample",
                    "MATCH (people:Person) RETURN people.name AS name LIMIT 10",
                ),
                (
                    "Top movies by actor count",
                    "MATCH (m:Movie)<-[:ACTED_IN]-(p:Person) "
                    "RETURN m.title AS movie, count(p) AS actors "
                    "ORDER BY actors DESC, movie ASC LIMIT 10",
                ),
                (
                    "People with sample movies",
                    "MATCH (p:Person)-[:ACTED_IN]->(m:Movie) "
                    "RETURN p.name AS person, collect(m.title)[0..3] AS movies "
                    "LIMIT 5",
                ),
            ]

            total_rows = 0
            for label, cypher in queries:
                records, _, _ = driver.execute_query(cypher)
                rows = [dict(record) for record in records]
                print(f"\n[Exercise 2.3] {label}: {len(rows)} row(s)")
                if rows:
                    print(pd.DataFrame(rows).to_string(index=False))
                    total_rows += len(rows)
                else:
                    print("No rows returned.")
    except Exception as exc:
        print(f"[Exercise 2.3] Neo4j connection/query failed: {exc}")
        print("Verify neo4j service is up and credentials are correct.")
        return

    if total_rows == 0:
        print(
            "\n[Exercise 2.3] no data matched the sample queries."
            " In Neo4j Browser, load a sample graph and rerun."
        )
        print("Tip: run ':play movie graph' in Browser to load demo data.")


def exercise_2_4():
    """Guide and validate the OpenSearch Dashboards workflow for exercise 2.4."""
    dashboards_url = "http://localhost:5601"
    opensearch_url = "http://localhost:9200"
    username = os.getenv("OPENSEARCH_USER", "admin").strip() or "admin"
    password = os.getenv("OPENSEARCH_PASSWORD", "@StrongP4ssword!")

    print(f"[Exercise 2.4] OpenSearch Dashboards URL: {dashboards_url}")
    print("[Exercise 2.4] credentials:")
    print(f"  user={username}")
    print(f"  password={password}")
    print("\n[Exercise 2.4] optional task: explore the web interface and sample data.")

    compose_path = Path("exercise_2_4/compose.yaml")
    if compose_path.is_file():
        print("\n[Exercise 2.4] if services are not running, start them with:")
        print("  docker compose -f exercise_2_4/compose.yaml up -d")
    else:
        print("\n[Exercise 2.4] warning: compose file not found at exercise_2_4/compose.yaml")

    try:
        with urllib.request.urlopen(dashboards_url, timeout=5) as resp:
            print(f"\n[Exercise 2.4] Dashboards reachable: HTTP {resp.status}")
    except Exception as exc:
        print(f"\n[Exercise 2.4] Dashboards check failed: {exc}")
        print("Start the stack, then open http://localhost:5601 in your browser.")
        return

    auth_raw = f"{username}:{password}".encode("utf-8")
    auth_header = b64encode(auth_raw).decode("ascii")
    request = urllib.request.Request(opensearch_url)
    request.add_header("Authorization", f"Basic {auth_header}")

    try:
        with urllib.request.urlopen(request, timeout=5) as resp:
            print(f"[Exercise 2.4] OpenSearch API reachable: HTTP {resp.status}")
    except urllib.error.HTTPError as exc:
        print(f"[Exercise 2.4] OpenSearch API check returned HTTP {exc.code}")
        print("If security is disabled in compose, this can still be acceptable.")
    except Exception as exc:
        print(f"[Exercise 2.4] OpenSearch API check failed: {exc}")
        print("Verify the OpenSearch container is running and port 9200 is exposed.")

    print("\n[Exercise 2.4] next steps in Dashboards:")
    print("1. Open http://localhost:5601.")
    print("2. Log in with admin / @StrongP4ssword! (if prompted).")
    print("3. Add and explore available sample data.")
    print("4. Build simple visualizations and dashboards.")


def _discover_exercises():
    """Discover exercise functions named like exercise_<chapter>_<number>."""
    discovered = {}
    for name, obj in globals().items():
        parts = name.split("_")
        if callable(obj) and len(parts) == 3 and parts[0] == "exercise":
            if parts[1].isdigit() and parts[2].isdigit():
                ex_id = f"{parts[1]}_{parts[2]}"
                discovered[ex_id] = obj

    def sort_key(ex_id):
        major, minor = ex_id.split("_")
        return int(major), int(minor)

    return dict(sorted(discovered.items(), key=lambda item: sort_key(item[0])))


def _normalize_exercise_token(token: str) -> str:
    """Normalize user-provided exercise IDs."""
    normalized = token.strip().lower().replace(".", "_").replace("-", "_")
    if normalized.isdigit():
        normalized = f"1_{normalized}"
    return normalized


def _parse_exercise_tokens(tokens, available_ids):
    """Parse exercise tokens and return a validated ordered selection."""
    available_set = set(available_ids)
    selected = []
    for token in tokens:
        normalized = _normalize_exercise_token(token)
        if normalized in {"all", "tutti", "*"}:
            return list(available_ids)
        if normalized not in available_set:
            valid = ", ".join(available_ids)
            raise ValueError(f"Unknown exercise '{token}'. Valid values: {valid}, tutti")
        if normalized not in selected:
            selected.append(normalized)
    return selected


def _prompt_exercise_selection(available_ids):
    """Prompt the user for exercise selection via terminal."""
    ids_display = ", ".join(available_ids)
    prompt = (
        f"\nWhich exercises do you want to run? ({ids_display} or 'all')\n"
        "You can enter multiple values separated by spaces or commas: "
    )
    while True:
        raw = input(prompt).strip()
        if not raw:
            print("Enter at least one exercise or 'all'.")
            continue
        tokens = raw.replace(",", " ").split()
        try:
            return _parse_exercise_tokens(tokens, available_ids)
        except ValueError as exc:
            print(exc)


if __name__ == "__main__":
    exercise_map = _discover_exercises()
    available_ids = list(exercise_map.keys())
    if not available_ids:
        raise SystemExit("No exercises found (expected functions like exercise_1_2).")

    parser = argparse.ArgumentParser(description="Run plotting exercises.")
    parser.add_argument(
        "-e",
        "--exercise",
        nargs="+",
        metavar="ID",
        help=(
            "Exercise ID to run (examples: 1_5, 1.5 or 5). "
            "You can pass multiple IDs, e.g. --exercise 1_3 1_5. "
            "Use 'all' to run all exercises. "
            "If omitted, an interactive prompt is shown."
        ),
    )
    args = parser.parse_args()

    if args.exercise:
        try:
            selected_ids = _parse_exercise_tokens(args.exercise, available_ids)
        except ValueError as exc:
            raise SystemExit(str(exc))
    else:
        selected_ids = _prompt_exercise_selection(available_ids)

    for ex_id in selected_ids:
        print(f"\n[Runner] running exercise {ex_id}")
        exercise_map[ex_id]()

