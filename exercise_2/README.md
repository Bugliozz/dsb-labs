# Exercise 2

This chapter contains the end-to-end mini data platform workflow built around the same source datasets.

## Local runner

Install chapter dependencies:

```bash
python -m pip install -r exercise_2/requirements.txt
```

Run the chapter-local launcher:

```bash
python exercise_2/main.py
```

Run specific steps:

```bash
python exercise_2/main.py --exercise 2_1
python exercise_2/main.py --exercise 2_1 2_3
python exercise_2/main.py --exercise all
```

## Stack bootstrap

```bash
docker network create network1
docker compose -f exercise_2/step_1/compose.yaml up -d
```

Service URLs:

- phpMyAdmin: `http://localhost:8080`
- Metabase: `http://localhost:3000`
- Neo4j Browser: `http://localhost:7474`
- OpenSearch Dashboards: `http://localhost:5601`

## Steps

- `2_1`: download Kaggle datasets, process `bankmarketing`, and upload MySQL tables
- `2_2`: validate MySQL reachability and print the Metabase setup checklist
- `2_3`: import MySQL primary tables into Neo4j
- `2_4`: import MySQL primary tables into OpenSearch

Use the root `main.py` only if you want the top-level launcher that dispatches across chapters.
