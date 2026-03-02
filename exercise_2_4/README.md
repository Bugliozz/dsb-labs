# Exercise 2.4 - OpenSearch (Optional)

This optional exercise focuses on OpenSearch Dashboards and sample data exploration.

## Start services

From project root:

```bash
docker compose -f exercise_2_4/compose.yaml up -d
```

## Access

- OpenSearch Dashboards: `http://localhost:5601`
- User: `admin`
- Password: `@StrongP4ssword!`

## What to do

1. Open Dashboards.
2. Log in if prompted.
3. Add sample data from the UI.
4. Explore visualizations and dashboards.

## Optional helper

```bash
python main.py --exercise 2_4
```

This prints the checklist and tries basic reachability checks for ports `5601` and `9200`.