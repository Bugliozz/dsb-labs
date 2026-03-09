# Exercise 2.1 - Kaggle to MySQL

Purpose: build the base relational dataset used by the rest of Exercise 2.

What it does:

1. Downloads datasets from Kaggle.
2. Processes the Bank Marketing dataset.
3. Loads tables into MySQL database `test`.

## Prerequisites

Start services:

```bash
docker network create network1
docker compose -f compose.yaml up -d
```

Kaggle credentials must be configured.

## Run

From project root:

```bash
python exercise_2/main.py --exercise 2_1
```

## Expected result

In MySQL (`test`) you should see:

- `bankmarketing`
- `livingwage50states` (if dataset download succeeds)

You can verify with phpMyAdmin at `http://localhost:8080`.
