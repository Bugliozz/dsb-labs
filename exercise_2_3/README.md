# Exercise 2.3 - Neo4j + Python

This exercise imports the MySQL `bankmarketing` table into Neo4j and runs Cypher analyses from Python.

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

1. Reads `test.bankmarketing` from MySQL.
2. Creates graph entities in Neo4j (`BankCustomer`, `Job`, `CampaignOutcome`, ...).
3. Executes analysis queries (outcome distribution, top jobs by positive rate, monthly contacts).

Optional row limit (default `5000`):

```bash
$env:NEO4J_IMPORT_LIMIT=10000
python main.py --exercise 2_3
```
