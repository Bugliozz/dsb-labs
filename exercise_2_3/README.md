# Exercise 2.3 - Neo4j + Python

This exercise imports MySQL primary tables (`bankmarketing`, `livingwage50states`) into Neo4j via Python.
Cypher analyses are run manually in Neo4j Browser.

## Prerequisites

1. Start services:

```bash
docker compose -f ../exercise_2_1/compose.yaml up -d
```

2. Open Neo4j Browser:

- URL: `http://localhost:7474`
- Username: `neo4j`
- Password: `test12345`

## Source data

Make sure Exercise 2.1 has loaded MySQL tables:

```bash
python main.py --exercise 2_1
```

## MySQL -> Neo4j import and query

From project root:

```bash
python main.py --exercise 2_3
```

What it does:

1. Reads `test.bankmarketing` and `test.livingwage50states` from MySQL.
2. Creates graph entities in Neo4j (`BankCustomer`, `Job`, `CampaignOutcome`, `LivingWageState`, ...).
3. Does not execute analytical Cypher queries automatically.
4. Ignores backup tables with suffix `_copy`.

Optional row limit (default `5000`):

```bash
$env:NEO4J_IMPORT_LIMIT=10000
python main.py --exercise 2_3
```

After import, open Neo4j Browser (`http://localhost:7474`) and run sample queries from the root `README.md` section "Exercise 2 Guide (Step by Step)".
