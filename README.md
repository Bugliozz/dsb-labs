# Data Science for Business Labs

A curated collection of Python labs developed for the **Data Science for Business** course.
The project combines data visualization, exploratory analysis, and Docker-based data tooling workflows.

## Portfolio Highlights

- Clean CLI runner (`main.py`) to execute one or multiple exercises.
- Reproducible visual analytics with `pandas`, `matplotlib`, and `seaborn`.
- Multi-service data stack for hands-on tooling practice (`MySQL`, `phpMyAdmin`, `Metabase`, `Neo4j`).
- Structured output folders (`datasets/`, `images/`) for each exercise.

## Tech Stack

- Python 3.10+
- pandas, NumPy, Matplotlib, Seaborn
- SQLAlchemy + MySQL connector
- Docker / Docker Compose
- KaggleHub (for selected datasets)

## Quick Start

```bash
git clone git@github.com:Bugliozz/exercise1.git
cd exercise1
python -m venv .venv
# Linux/macOS
source .venv/bin/activate
# Windows PowerShell
# .venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Run the CLI:

```bash
python main.py
```

Run a specific exercise:

```bash
python main.py --exercise 1_5
```

Run multiple exercises:

```bash
python main.py --exercise 1_3 1_5
```

Run all:

```bash
python main.py --exercise all
```

## Exercises

- `exercise_1_2`: multi-panel plotting layout.
- `exercise_1_3`: composed Seaborn figure (line/scatter/bar/heatmap).
- `exercise_1_4`: EDA on Indian Liver Patient Records.
- `exercise_1_5`: ETF correlation analysis (heatmap + ellipse matrix).
- `exercise_1_6`: Docker + DokuWiki lab scaffold.
- `exercise_1_7`: Flask + Docker lab scaffold.
- `exercise_2_1`: Kaggle ingestion + MySQL upload workflow.

## Project Structure

```text
exercise1/
  main.py
  requirements.txt
  exercise_1_2/
  exercise_1_3/
  exercise_1_4/
  exercise_1_5/
  exercise_1_6/
  exercise_1_7/
  exercise_2_1/
```

## Data Notes

- Some datasets are expected locally for reproducibility.
- Kaggle-based steps require valid Kaggle credentials configured in your environment.
- Generated images are included where useful to document outputs.

## Exercise 2.1 Services

Start the service stack:

```bash
docker compose -f exercise_2_1/compose.yaml up -d
```

Available interfaces:

- phpMyAdmin: `http://localhost:8080`
- Metabase: `http://localhost:3000`
- Neo4j: `http://localhost:7474`

Then run:

```bash
python main.py --exercise 2_1
```
