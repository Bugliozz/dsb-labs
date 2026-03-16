# Exercise 3

This folder contains all assets related to the machine-learning exercise on Ames house-price prediction.

## Structure

- `main.py`: local runner for Exercise 3 only
- `requirements.txt`: dependencies for Exercise 3
- `tests/`: tests scoped to Exercise 3
- `step_1/`: training workflow, artifacts, API app, Docker image

## Commands

You can reach Exercise 3 either from the root launcher:

```bash
python main.py --exercise 3_1 --command train
python main.py --exercise 3_1 --command serve
```

Or directly from the local runner:

Install only Exercise 3 dependencies:

```bash
pip install -r exercise_3/requirements.txt
```

Run the offline training workflow:

```bash
python exercise_3/main.py train
```

Start the local Flask API:

```bash
python exercise_3/main.py serve
```

Run Exercise 3 tests:

```bash
python -m unittest exercise_3.tests.test_step_1.AmesApiTestCase
```

## Cloud Run

The inference API is also deployed on Cloud Run.

- service: `ames-house-price-api`
- region: `europe-west1`
- verified on `2026-03-16`
- URL: `https://ames-house-price-api-47880508774.europe-west1.run.app`
