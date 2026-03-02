# Data Science for Business Lab - Exercise 1

## Setup

Clone the repository and install dependencies:

```bash
git clone git@github.com:mb-uninsubria/exercise1.git
cd exercise1
python -m venv env           # or another virtual environment tool
# on Unix/macOS
source env/bin/activate
# on Windows PowerShell
# env\Scripts\Activate.ps1
pip install -r requirements.txt
```

Run the script:

```bash
python main.py
```

When launched without arguments, the terminal asks which exercise(s) to run
(for example `1_2`, `1_3`, or `tutti`).

Run a specific exercise only:

```bash
python main.py --exercise 1_5
```

Run multiple selected exercises:

```bash
python main.py --exercise 1_3 1_5
```

Run all exercises explicitly:

```bash
python main.py --exercise tutti
```

## Project structure

```text
exercise1/
  main.py
  requirements.txt
  exercise_1_2/
    images/
      exercise1_plots.png
    datasets/
  exercise_1_3/
    images/
      exercise1_3.png
    datasets/
  exercise_1_4/
    images/
      exercise1_4_*.png
    datasets/
      indian_liver_patient.csv
  exercise_1_5/
    images/
      exercise1_5_etf_correlation_*.png
    datasets/
      sp500_data.csv.gz
      sp500_sectors.csv
  exercise_1_6/
    docker-compose.yml
    README.md
    dokuwiki_data/
  exercise_1_7/
    app.py
    requirements.txt
    Dockerfile
    README.md
```

## Exercise 1.2

`exercise_1_2()` creates the composite 2x4 figure and saves:

- `exercise_1_2/images/exercise1_plots.png`

## Exercise 1.3

`exercise_1_3()` creates the 2x2 Seaborn figure and saves:

- `exercise_1_3/images/exercise1_3.png`

## Optional Exercise 1.4

For the Indian Liver Patient Records dataset:

1. Download the CSV from:
   https://www.kaggle.com/datasets/uciml/indian-liver-patient-records
2. Put the file in:
   `exercise_1_4/datasets/indian_liver_patient.csv`
3. Run `python main.py`.

Generated charts are saved in:

- `exercise_1_4/images/`

## Exercise 1.5 (PS4DS Chapter 1 replica)

`exercise_1_5()` replicates the ETF correlation visualizations from:

- https://github.com/gedeck/practical-statistics-for-data-scientists
- Notebook: `python/notebooks/Chapter 1 - Exploratory Data Analysis.ipynb`

Datasets used:

- `exercise_1_5/datasets/sp500_data.csv.gz`
- `exercise_1_5/datasets/sp500_sectors.csv`

Generated charts:

- `exercise_1_5/images/exercise1_5_etf_correlation_heatmap.png`
- `exercise_1_5/images/exercise1_5_etf_correlation_ellipses.png`

## Exercise 1.6 (Docker + DokuWiki)

`exercise_1_6()` creates a complete Docker lab scaffold:

- `exercise_1_6/docker-compose.yml`
- `exercise_1_6/README.md`
- `exercise_1_6/dokuwiki_data/` (persistent storage)

Run:

```bash
python main.py --exercise 1_6
```

Then:

```bash
cd exercise_1_6
docker compose up -d
```

Open `http://localhost:8080/`.

## Exercise 1.7 (Flask + Docker)

`exercise_1_7()` creates a dedicated Flask + Docker lab scaffold:

- `exercise_1_7/app.py`
- `exercise_1_7/requirements.txt`
- `exercise_1_7/Dockerfile`
- `exercise_1_7/README.md`

Run:

```bash
python main.py --exercise 1_7
```

Then:

```bash
cd exercise_1_7
docker build --tag python-docker .
docker run -d -p 5000:5000 --name python-docker-lab python-docker
```

Open `http://localhost:5000/`.
