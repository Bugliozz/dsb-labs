# Exercise 2.2 - Metabase + MySQL

This exercise focuses on connecting Metabase to the MySQL container and exploring tables loaded in Exercise 2.1.

## Prerequisites

1. Start services:

```bash
docker compose -f ../exercise_2_1/compose.yaml up -d
```

2. Ensure tables are present (from project root):

```bash
python main.py --exercise 2_1
```

## Metabase setup

Open: `http://localhost:3000`

When adding MySQL in Metabase, use:

- Host: `db`
- Port: `3306`
- Database: `test`
- Username: `root`
- Password: `pass`

After saving, use **Browse data** to explore tables and build questions/charts.

## Optional helper

From project root:

```bash
python main.py --exercise 2_2
```

This validates host-side MySQL reachability and prints the setup checklist.