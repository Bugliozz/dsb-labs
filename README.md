# Data Science for Business Labs

A curated collection of Python labs developed for the **Data Science for Business** course.
The project combines data visualization, exploratory analysis, and Docker-based data tooling workflows.

## Project Highlights

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
- `exercise_2_2`: Metabase login + MySQL connection and data exploration workflow.
- `exercise_2_3`: MySQL -> Neo4j graph import and Cypher analysis workflow.
- `exercise_2_4`: MySQL -> OpenSearch import and Dashboards visualization workflow.

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
  exercise_2_2/
  exercise_2_3/
  exercise_2_4/
```

## Data Notes

- Some datasets are expected locally for reproducibility.
- Kaggle-based steps require valid Kaggle credentials configured in your environment.
- Generated images are included where useful to document outputs.

## Exercise 1 Guide (Step by Step)

### Goal

Exercise 1 is an introductory path:

1. first exercises focus on plotting and visual exploration;
2. last exercises introduce Docker-based application/container workflows.

### Intro part: charts and EDA

These exercises are mainly for learning how to create and read plots:

- `1_2`: multi-panel matplotlib layout;
- `1_3`: combined Seaborn charts;
- `1_4`: exploratory analysis on a medical dataset;
- `1_5`: correlation analysis and visual diagnostics.

Run examples:

```bash
python main.py --exercise 1_2
python main.py --exercise 1_3
python main.py --exercise 1_4
python main.py --exercise 1_5
```

### Final part: Docker introduction

These exercises shift focus from plotting to tooling and deployment basics:

- `1_6`: DokuWiki with Docker Compose and persistent volumes;
- `1_7`: minimal Flask app containerization with Dockerfile.

Run examples:

```bash
python main.py --exercise 1_6
python main.py --exercise 1_7
```

### Full Exercise 1 sequence

```bash
python main.py --exercise 1_2 1_3 1_4 1_5 1_6 1_7
```

## Exercise 2 Guide (Step by Step)

### Goal

Exercise 2 is an end-to-end mini data platform workflow:

1. ingest data into MySQL (`2_1`);
2. explore tabular analytics in Metabase (`2_2`);
3. reshape the same data as graph data in Neo4j (`2_3`);
4. index the same data for search/BI in OpenSearch (`2_4`).

### 0. Prerequisites

From project root:

```powershell
# Reset complete venv (if active)
deactivate 2>$null

# Recreate clean environment
Remove-Item -Recurse -Force .venv
python -m venv .venv
.venv\Scripts\Activate.ps1

# Reinstall dependencies
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

# Quick verification
python -c "import sys, numpy; print(sys.executable); print(numpy.__version__)"

```

Kaggle credentials are required for `2_1`:

- `C:\Users\<you>\.kaggle\kaggle.json`, or
- env vars `KAGGLE_USERNAME` and `KAGGLE_KEY`.

### 1. Start base services (MySQL, phpMyAdmin, Metabase, Neo4j)

```bash
docker network create network1
docker compose -f exercise_2_1/compose.yaml up -d
```

Service URLs:

- phpMyAdmin: `http://localhost:8080`
- Metabase: `http://localhost:3000`
- Neo4j Browser: `http://localhost:7474`

### 2. Run Exercise 2.1 (Kaggle -> MySQL)

What it does:

- downloads datasets from Kaggle;
- processes `bankmarketing`;
- writes tables into MySQL database `test`.

Run:

```bash
python main.py --exercise 2_1
```
phpMyAdmin credentials:

- Server: `db` 
- Username: `root`
- Password: `pass`

Expected MySQL tables:

- `bankmarketing`
- `livingwage50states` (if dataset available)

Verifica manuale in phpMyAdmin:

1. Apri `http://localhost:8080`.
2. Accedi con `root / pass`.
3. Seleziona database `test` e apri tab `SQL`.
4. Esegui query di esempio:

```sql
SELECT y AS outcome, COUNT(*) AS customers
FROM bankmarketing
GROUP BY y
ORDER BY customers DESC;
```

Ritorna il numero di clienti per outcome della campagna (`y`, tipicamente `yes/no`).

```sql
SELECT state_territory, oneadult_nokids
FROM livingwage50states
ORDER BY oneadult_nokids DESC
LIMIT 10;
```

Ritorna i 10 stati con living wage oraria piu alta per profilo `oneadult_nokids`.

### 3. Run Exercise 2.2 (MySQL -> Metabase)

What it does:

- validates MySQL reachability from host;
- prints setup checklist for Metabase;
- keeps the workflow focused on primary tables (`bankmarketing`, `livingwage50states`);
- ignores backup tables with suffix `_copy`.

Run:

```bash
python main.py --exercise 2_2
```

Metabase connection fields:

- Host: `db`
- Port: `3306`
- Database: `test`
- Username: `root`
- Password: `pass`

Common issue: `RSA public key is not available`

- In Metabase advanced connection options use JDBC params:
  `allowPublicKeyRetrieval=true&useSSL=false`
- Or create a dedicated MySQL user with `mysql_native_password`.

Query manuali in Metabase:

1. Apri `http://localhost:3000`.
2. Vai su **New -> SQL query** e seleziona il DB MySQL `test`.
3. Esegui query di esempio:

```sql
SELECT y AS outcome, COUNT(*) AS customers
FROM bankmarketing
GROUP BY y
ORDER BY customers DESC;
```

Ritorna la distribuzione degli outcome marketing.

```sql
SELECT job, ROUND(AVG(balance), 2) AS avg_balance, COUNT(*) AS total_customers
FROM bankmarketing
GROUP BY job
ORDER BY avg_balance DESC
LIMIT 10;
```

Ritorna i job con saldo medio piu alto e numero record associati.

```sql
SELECT state_territory, oneadult_nokids
FROM livingwage50states
ORDER BY oneadult_nokids DESC
LIMIT 10;
```

Ritorna i 10 stati con living wage oraria piu alta per un adulto senza figli.

### 4. Run Exercise 2.3 (MySQL -> Neo4j)

What it does:

- reads MySQL primary tables `test.bankmarketing` and `test.livingwage50states`;
- imports them into Neo4j graph entities (`BankCustomer`, `Job`, `CampaignOutcome`, `LivingWageState`, ...);
- does not run analytical Cypher queries automatically;
- ignores backup tables with suffix `_copy`.

Run:

```bash
python main.py --exercise 2_3
```

Neo4j credentials:

- URL: `http://localhost:7474` (browser)
- Bolt URI used by Python: `neo4j://localhost:7687`
- Username: `neo4j`
- Password: `test12345`

Optional tuning:

- `NEO4J_IMPORT_LIMIT` (default `5000`)

Query manuali in Neo4j Browser:

1. Apri `http://localhost:7474`.
2. Accedi con `neo4j / test12345`.
3. Esegui query nella barra Cypher:

```cypher
MATCH (c:BankCustomer)-[:HAS_OUTCOME]->(o:CampaignOutcome)
RETURN o.name AS outcome, count(*) AS customers
ORDER BY customers DESC, outcome ASC;
```

Ritorna la distribuzione degli outcome della campagna nel grafo.

```cypher
MATCH (c:BankCustomer)-[:HAS_JOB]->(j:Job)
MATCH (c)-[:HAS_OUTCOME]->(o:CampaignOutcome)
WITH j.name AS job, count(*) AS total,
     sum(CASE WHEN toLower(toString(o.name)) IN ['1','yes','true'] THEN 1 ELSE 0 END) AS positives
RETURN job, total, round(100.0 * positives / total, 2) AS positive_rate_pct
ORDER BY positive_rate_pct DESC, total DESC
LIMIT 10;
```

Ritorna i job con tasso di risposta positiva piu alto.

```cypher
MATCH (s:LivingWageState)-[r:HAS_LIVING_WAGE]->(h:HouseholdProfile {name:'oneadult_nokids'})
RETURN s.name AS state, r.hourly_wage AS oneadult_nokids_wage
ORDER BY oneadult_nokids_wage DESC
LIMIT 10;
```

Ritorna i 10 stati con living wage piu alta per profilo `oneadult_nokids`.

### 5. Run Exercise 2.4 (MySQL -> OpenSearch)

Start OpenSearch stack:

```bash
docker compose -f exercise_2_4/compose.yaml up -d
```

What it does:

- reads MySQL primary tables `test.bankmarketing` and `test.livingwage50states`;
- bulk indexes both into OpenSearch (`bankmarketing` and `livingwage50states` by default);
- does not run analytical OpenSearch queries automatically;
- guides visualization in Dashboards;
- ignores backup tables with suffix `_copy`.

Run:

```bash
python main.py --exercise 2_4
```

OpenSearch credentials:

- Dashboards URL: `http://localhost:5601`
- User: `admin`
- Password: `@StrongP4ssword!`

Optional tuning:

- `OPENSEARCH_BANK_INDEX` (default `bankmarketing`)
- `OPENSEARCH_LIVINGWAGE_INDEX` (default `livingwage50states`)
- `OPENSEARCH_IMPORT_LIMIT` (default `5000`)

Query manuali in OpenSearch Dashboards (Dev Tools -> Console):

1. Apri `http://localhost:5601`.
2. Login con `admin / @StrongP4ssword!`.
3. Vai in **Dev Tools -> Console**.
4. Esegui query di esempio:

```http
GET bankmarketing/_count
```

Ritorna il numero di documenti indicizzati nell'indice `bankmarketing`.

```http
POST bankmarketing/_search
{
  "size": 0,
  "aggs": {
    "by_outcome": {
      "terms": {
        "field": "y",
        "size": 10
      }
    }
  }
}
```

Ritorna una aggregazione con la distribuzione dei valori `y` (outcome campagna).

```http
POST livingwage50states/_search
{
  "size": 10,
  "_source": ["state_territory", "oneadult_nokids"],
  "sort": [
    {
      "oneadult_nokids": {
        "order": "desc",
        "missing": "_last"
      }
    }
  ]
}
```

Ritorna i 10 documenti/stati con living wage `oneadult_nokids` piu alta.

### 6. Full run sequence

```bash
python main.py --exercise 2_1
python main.py --exercise 2_2
python main.py --exercise 2_3
docker compose -f exercise_2_4/compose.yaml up -d
python main.py --exercise 2_4
```

### 7. Shutdown

```bash
docker compose -f exercise_2_1/compose.yaml down
docker compose -f exercise_2_4/compose.yaml down
```
