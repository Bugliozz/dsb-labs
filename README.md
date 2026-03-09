# Data Science for Business Labs

Repository for the **Data Science for Business** course labs.

The root `main.py` is now a lightweight launcher: it asks which chapter and step to run, then dispatches execution to the local runner inside `exercise_1/`, `exercise_2/`, or `exercise_3/`.

## Project layout

```text
dsb-labs/
  main.py
  lab_catalog.py
  requirements.txt
  tests/
  exercise_1/
    main.py
    requirements.txt
    README.md
    tests/
    step_2/
    step_3/
    step_4/
    step_5/
    step_6/
    step_7/
  exercise_2/
    main.py
    requirements.txt
    README.md
    tests/
    step_1/
    step_2/
    step_3/
    step_4/
  exercise_3/
    main.py
    requirements.txt
    README.md
    tests/
    step_1/
```

Note: `exercise_1` starts from `step_2` to match the original course numbering.

## Installation

Install everything from the repo root:

```bash
python -m pip install -r requirements.txt
```

Or install only a chapter:

```bash
python -m pip install -r exercise_1/requirements.txt
python -m pip install -r exercise_2/requirements.txt
python -m pip install -r exercise_3/requirements.txt
```

## Root launcher

Interactive mode:

```bash
python main.py
```

Examples:

```bash
python main.py --chapter 2 --exercise 2_1 2_3
python main.py --chapter 2 --exercise all
python main.py --exercise 1_2 2_4
python main.py --exercise 3_1 --command train
python main.py --exercise 3_1 --command serve
```

Rules:

- `all` is valid only together with `--chapter`
- `Exercise 3` needs `--command train|serve` in non-interactive mode
- the root launcher uses `subprocess` and does not import the heavy dependencies of unselected chapters

## Local runners

```bash
python exercise_1/main.py
python exercise_2/main.py
python exercise_3/main.py train
python exercise_3/main.py serve
```

Direct step selection:

```bash
python exercise_1/main.py --exercise 1_2 1_5
python exercise_2/main.py --exercise 2_1 2_4
python exercise_3/main.py train --exercise 3_1
```

## Chapter overview

### Exercise 1

- `1_2`: multi-panel matplotlib layout
- `1_3`: composed seaborn charts
- `1_4`: EDA on Indian Liver Patient Records
- `1_5`: ETF correlation analysis
- `1_6`: Docker + DokuWiki scaffold validation
- `1_7`: Flask + Docker scaffold validation

See [exercise_1/README.md](/c:/Users/marco/OneDrive%20-%20Universit%C3%A0%20degli%20Studi%20dell'Insubria/Universit%C3%A0/Magistrale/DATA%20SCIENCE%20FOR%20BUSINESS/dsb-labs/exercise_1/README.md).

### Exercise 2

Exercise 2 is the end-to-end mini data platform workflow:

1. ingest data into MySQL (`2_1`)
2. explore tables in Metabase (`2_2`)
3. reshape them in Neo4j (`2_3`)
4. index them in OpenSearch (`2_4`)

Stack bootstrap:

```bash
docker network create network1
docker compose -f exercise_2/step_1/compose.yaml up -d
```

Service URLs:

- phpMyAdmin: `http://localhost:8080`
- Metabase: `http://localhost:3000`
- Neo4j Browser: `http://localhost:7474`
- OpenSearch Dashboards: `http://localhost:5601`

See [exercise_2/README.md](/c:/Users/marco/OneDrive%20-%20Universit%C3%A0%20degli%20Studi%20dell'Insubria/Universit%C3%A0/Magistrale/DATA%20SCIENCE%20FOR%20BUSINESS/dsb-labs/exercise_2/README.md).

### Exercise 3

Exercise 3 contains the Ames house-price workflow and Flask API:

- `3_1` with `train`: build plots, train models, save artifacts
- `3_1` with `serve`: start the local inference app

See [exercise_3/README.md](/c:/Users/marco/OneDrive%20-%20Universit%C3%A0%20degli%20Studi%20dell'Insubria/Universit%C3%A0/Magistrale/DATA%20SCIENCE%20FOR%20BUSINESS/dsb-labs/exercise_3/README.md).

## Tests

Runner-level tests:

```bash
python -m unittest tests.test_main_launcher
python -m unittest exercise_1.tests.test_runner
python -m unittest exercise_2.tests.test_runner
python -m unittest exercise_3.tests.test_step_1
```
