# Exercise 2.3 - Neo4j + Python

This exercise focuses on Neo4j Browser tutorials and reproducing sample Cypher queries from Python.

## Prerequisites

1. Start services:

```bash
docker compose -f ../exercise_2_1/compose.yaml up -d
```

2. Open Neo4j Browser:

- URL: `http://localhost:7474`
- Username: `neo4j`
- Password: `test12345`

## Browser tasks

Complete these tutorials from the homepage:

1. `Getting started with Neo4j Browser`
2. `Try Neo4j with live data`
3. `Cypher basics`

## Python replication

From project root:

```bash
python main.py --exercise 2_3
```

The command checks connectivity and runs sample Cypher queries such as:

- `MATCH (people:Person) RETURN people.name LIMIT 10`
- Movie/actor aggregation examples.

If no rows are returned, load demo data in Browser first (example: `:play movie graph`).