# Exercise 2.4 - OpenSearch (Optional)

This optional exercise imports MySQL primary tables (`bankmarketing`, `livingwage50states`) into OpenSearch and visualizes them in Dashboards.

## Start services

From project root:

```bash
docker compose -f exercise_2_1/compose.yaml up -d
```

## Access

- OpenSearch Dashboards: `http://localhost:5601`
- User: `admin`
- Password: `@StrongP4ssword!`

## Source data

Load MySQL tables first:

```bash
python main.py --exercise 2_1
```

## MySQL -> OpenSearch import

```bash
python main.py --exercise 2_4
```

Then in Dashboards:

1. Open Dashboards.
2. Log in if prompted.
3. Create Data Views for indexes `bankmarketing` and `livingwage50states`.
4. Open Dev Tools -> Console and run sample queries (see root `README.md`).
5. Use Discover and Dashboard to explore both datasets.
6. Ignore backup tables with suffix `_copy`.

## Optional helper

```bash
python main.py --exercise 2_4
```

Optional tuning:

```bash
$env:OPENSEARCH_IMPORT_LIMIT=10000
$env:OPENSEARCH_BANK_INDEX=bankmarketing_v2
$env:OPENSEARCH_LIVINGWAGE_INDEX=livingwage50states_v2
python main.py --exercise 2_4
```
