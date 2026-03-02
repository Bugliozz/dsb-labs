# Exercise 2.4 - OpenSearch (Optional)

This optional exercise imports MySQL `bankmarketing` data into OpenSearch and visualizes it in Dashboards.

## Start services

From project root:

```bash
docker compose -f exercise_2_4/compose.yaml up -d
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
3. Create a Data View for index `bankmarketing` (or your `OPENSEARCH_INDEX`).
4. Use Discover and Dashboard to explore fields (`y`, `job`, `marital`, `month`, ...).

## Optional helper

```bash
python main.py --exercise 2_4
```

Optional tuning:

```bash
$env:OPENSEARCH_IMPORT_LIMIT=10000
$env:OPENSEARCH_INDEX=bankmarketing_v2
python main.py --exercise 2_4
```
