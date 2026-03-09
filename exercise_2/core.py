"""Exercise 2 data platform workflows."""

from __future__ import annotations

import json
import os
import shutil
import urllib.error
import urllib.request
from base64 import b64encode
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
STEP_1_DIR = BASE_DIR / "step_1"


def _read_csv_with_auto_separator(csv_path: Path) -> tuple[pd.DataFrame, str]:
    """Read a CSV trying comma first and semicolon as fallback."""
    df = pd.read_csv(csv_path)
    if len(df.columns) == 1:
        df = pd.read_csv(csv_path, sep=";")
        return df, ";"
    return df, ","


def _convert_text_binary_columns(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, dict[str, int]]]:
    """Convert binary text columns (yes/no etc.) to integer 0/1."""
    converted: dict[str, dict[str, int]] = {}
    out = df.copy()
    for col in out.columns:
        if not (pd.api.types.is_object_dtype(out[col]) or pd.api.types.is_string_dtype(out[col])):
            continue

        normalized = out[col].dropna().astype(str).str.strip().str.lower()
        unique_vals = set(normalized.unique())
        mapping = None
        if unique_vals == {"yes", "no"}:
            mapping = {"no": 0, "yes": 1}
        elif unique_vals == {"true", "false"}:
            mapping = {"false": 0, "true": 1}
        elif unique_vals == {"y", "n"}:
            mapping = {"n": 0, "y": 1}

        if mapping is None:
            continue

        out[col] = out[col].astype(str).str.strip().str.lower().map(mapping).astype("Int64")
        converted[col] = mapping

    return out, converted


def _download_dataset_csvs(dataset_id: str, target_dir: Path, kagglehub_module) -> list[Path]:
    """Download a Kaggle dataset and copy all CSV files to target_dir."""
    cache_path = Path(kagglehub_module.dataset_download(dataset_id))
    csv_files = sorted(cache_path.rglob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found for dataset '{dataset_id}'")

    target_dir.mkdir(parents=True, exist_ok=True)
    copied_files = []
    for src in csv_files:
        dest = target_dir / src.name
        shutil.copy2(src, dest)
        copied_files.append(dest)
    return copied_files


def _existing_csvs(target_dir: Path) -> list[Path]:
    """Return existing CSV files in target_dir."""
    if not target_dir.is_dir():
        return []
    return sorted(target_dir.rglob("*.csv"))


def _ensure_dataset_csvs(
    dataset_id: str,
    target_dir: Path,
    kagglehub_module,
    force_download: bool = False,
) -> list[Path]:
    """Reuse local CSVs when available, otherwise download from Kaggle."""
    existing_files = _existing_csvs(target_dir)
    if existing_files and not force_download:
        return existing_files
    return _download_dataset_csvs(dataset_id, target_dir, kagglehub_module)


def _pick_bank_marketing_csv(csv_files: list[Path]) -> Path:
    """Pick the most likely Bank Marketing CSV from a list of files."""
    for csv_file in csv_files:
        name = csv_file.stem.lower()
        if "bank" in name and "marketing" in name:
            return csv_file
    for csv_file in csv_files:
        if "bank" in csv_file.stem.lower():
            return csv_file
    return csv_files[0]


def exercise_2_1() -> None:
    """Run the full Kaggle + MySQL workflow for exercise 2.1."""
    compose_path = STEP_1_DIR / "compose.yaml"
    dataset_dir = STEP_1_DIR / "datasets"
    living_wage_dir = dataset_dir / "living_wage_50_states"
    bank_marketing_dir = dataset_dir / "bank_marketing"
    dataset_dir.mkdir(parents=True, exist_ok=True)

    if compose_path.is_file():
        print(f"[Exercise 2.1] using compose file: {compose_path}")
        print("[Exercise 2.1] start services with: docker compose -f exercise_2/step_1/compose.yaml up -d")
    else:
        print(f"[Exercise 2.1] warning: compose file not found at {compose_path}")

    try:
        import kagglehub
    except ImportError:
        print("[Exercise 2.1] kagglehub is not installed.")
        print("Install it with: python -m pip install -r exercise_2/requirements.txt")
        return

    force_download = os.getenv("FORCE_KAGGLE_DOWNLOAD", "0").strip().lower() in {"1", "true", "yes", "y"}

    living_wage_id = "brandonconrady/living-wage-50-states"
    print(f"[Exercise 2.1] downloading dataset: {living_wage_id}")
    try:
        living_wage_csvs = _ensure_dataset_csvs(
            living_wage_id,
            living_wage_dir,
            kagglehub,
            force_download=force_download,
        )
        print(f"[Exercise 2.1] {len(living_wage_csvs)} CSV file(s) ready in {living_wage_dir}")
    except Exception as exc:
        living_wage_csvs = []
        print(f"[Exercise 2.1] living wage download failed: {exc}")
        print("Make sure your Kaggle credentials are configured correctly.")

    bank_dataset_candidates = [
        "fedesoriano/bank-marketing",
        "janiobachmann/bank-marketing-dataset",
        "rouseguy/bankbalanced",
    ]
    custom_bank_dataset = os.getenv("BANK_MARKETING_DATASET", "").strip()
    if custom_bank_dataset:
        print(f"[Exercise 2.1] custom Bank Marketing dataset from env: {custom_bank_dataset}")
        bank_dataset_candidates.insert(0, custom_bank_dataset)

    bank_csvs = _existing_csvs(bank_marketing_dir) if not force_download else []
    used_bank_dataset = "local cache" if bank_csvs else None
    if not bank_csvs:
        for dataset_id in bank_dataset_candidates:
            print(f"[Exercise 2.1] trying Bank Marketing dataset: {dataset_id}")
            try:
                bank_csvs = _download_dataset_csvs(dataset_id, bank_marketing_dir, kagglehub)
                used_bank_dataset = dataset_id
                break
            except Exception as exc:
                print(f"[Exercise 2.1] failed for {dataset_id}: {exc}")

    if not bank_csvs:
        print("[Exercise 2.1] unable to download a Bank Marketing dataset.")
        print("Set BANK_MARKETING_DATASET with a valid Kaggle dataset ID and run again.")
        return

    print(f"[Exercise 2.1] using Bank Marketing dataset: {used_bank_dataset}")
    bank_csv_path = _pick_bank_marketing_csv(bank_csvs)
    bank_df, sep_used = _read_csv_with_auto_separator(bank_csv_path)
    print(
        f"[Exercise 2.1] loaded {bank_csv_path.name} ({len(bank_df)} rows x {len(bank_df.columns)} columns, sep='{sep_used}')"
    )

    converted_df, converted_map = _convert_text_binary_columns(bank_df)
    if converted_map:
        print("[Exercise 2.1] converted binary text columns:")
        for col, mapping in converted_map.items():
            print(f"  - {col}: {mapping}")
    else:
        print("[Exercise 2.1] no text binary columns converted.")

    processed_csv = bank_marketing_dir / "bank_marketing_processed.csv"
    converted_df.to_csv(processed_csv, index=False)
    print(f"[Exercise 2.1] saved processed CSV: {processed_csv}")

    try:
        from sqlalchemy import create_engine, text
    except ImportError:
        print("[Exercise 2.1] SQLAlchemy is not installed.")
        print("Install with: python -m pip install -r exercise_2/requirements.txt")
        return

    db_user = os.getenv("MYSQL_USER", "root").strip() or "root"
    db_password = os.getenv("MYSQL_PASSWORD", "pass")
    db_host = os.getenv("MYSQL_HOST", "localhost").strip() or "localhost"
    db_port = int(os.getenv("MYSQL_PORT", "3306").strip() or "3306")
    db_name = os.getenv("MYSQL_DATABASE", "test").strip() or "test"

    print(f"[Exercise 2.1] MySQL target: host={db_host} port={db_port} db={db_name} user={db_user}")

    server_url = f"mysql+mysqlconnector://{db_user}:{db_password}@{db_host}:{db_port}"
    db_url = f"{server_url}/{db_name}"
    try:
        server_engine = create_engine(server_url, echo=False)
        with server_engine.connect() as conn:
            conn.execute(text(f"CREATE DATABASE IF NOT EXISTS `{db_name}`"))
            conn.commit()
        server_engine.dispose()

        engine = create_engine(db_url, echo=False)
        converted_df.to_sql(name="bankmarketing", con=engine, if_exists="replace", index=False)
        print("[Exercise 2.1] uploaded table: bankmarketing")

        converted_df.to_sql(name="bankmarketing_copy", con=engine, if_exists="replace", index=False)
        print("[Exercise 2.1] created table copy: bankmarketing_copy")

        if living_wage_csvs:
            living_wage_path = living_wage_csvs[0]
            living_wage_df, _ = _read_csv_with_auto_separator(living_wage_path)
            living_wage_df.to_sql(name="livingwage50states", con=engine, if_exists="replace", index=False)
            print("[Exercise 2.1] uploaded table: livingwage50states")
            living_wage_df.to_sql(name="livingwage50states_copy", con=engine, if_exists="replace", index=False)
            print("[Exercise 2.1] created table copy: livingwage50states_copy")

        print("[Exercise 2.1] open phpMyAdmin (http://localhost:8080) for manual SQL checks.")
        engine.dispose()
    except Exception as exc:
        print(f"[Exercise 2.1] MySQL upload failed: {exc}")
        print("Verify Docker services are running and phpMyAdmin can access db/root/pass.")


def exercise_2_2() -> None:
    """Guide and validate the Metabase + MySQL workflow for exercise 2.2."""
    print("[Exercise 2.2] Metabase URL: http://localhost:3000")
    print("[Exercise 2.2] target MySQL connection:")
    print("  host=db")
    print("  port=3306")
    print("  database=test")
    print("  username=root")
    print("  password=pass")

    compose_path = STEP_1_DIR / "compose.yaml"
    if compose_path.is_file():
        print("\n[Exercise 2.2] if services are not running, start them with:")
        print("  docker compose -f exercise_2/step_1/compose.yaml up -d")
    else:
        print("\n[Exercise 2.2] warning: compose file not found at exercise_2/step_1/compose.yaml")

    try:
        from sqlalchemy import create_engine, text
    except ImportError:
        print("\n[Exercise 2.2] SQLAlchemy not found.")
        print("Install with: python -m pip install -r exercise_2/requirements.txt")
        return

    host_db_url = "mysql+mysqlconnector://root:pass@localhost:3306/test"
    try:
        engine = create_engine(host_db_url, echo=False)
        with engine.connect() as conn:
            tables = conn.execute(text("SHOW TABLES")).fetchall()
        engine.dispose()
    except Exception as exc:
        print(f"\n[Exercise 2.2] MySQL host-side check failed: {exc}")
        print("Make sure Docker services are up and MySQL is reachable on localhost:3306.")
        return

    table_names = [row[0] for row in tables]
    primary_tables = ["bankmarketing", "livingwage50states"]
    available_primary = [name for name in primary_tables if name in table_names]
    backup_tables = sorted(name for name in table_names if name.endswith("_copy"))

    print(f"\n[Exercise 2.2] MySQL reachable from host. Tables found: {len(table_names)} total")
    if available_primary:
        print("[Exercise 2.2] primary tables used from this point onward:")
        for name in available_primary:
            print(f"  - {name}")
    else:
        print("[Exercise 2.2] primary tables not found. Run: python exercise_2/main.py --exercise 2_1")

    if backup_tables:
        print(f"[Exercise 2.2] backup tables detected ({len(backup_tables)}) and ignored:")
        for name in backup_tables[:10]:
            print(f"  - {name}")
        if len(backup_tables) > 10:
            print(f"  ... and {len(backup_tables) - 10} more")

    print("\n[Exercise 2.2] Metabase setup checklist:")
    print("1. Open http://localhost:3000 and complete initial admin login.")
    print("2. Add database -> MySQL.")
    print("3. Use host=db, port=3306, db=test, username=root, password=pass.")
    print("4. Save and wait for sync.")
    print("5. Explore only bankmarketing and livingwage50states in 'Browse data'.")
    print("6. Ignore backup tables with suffix '_copy'.")


def exercise_2_3() -> None:
    """Load MySQL primary tables into Neo4j and guide manual Cypher analyses."""
    print("[Exercise 2.3] Neo4j Browser URL: http://localhost:7474")
    print("[Exercise 2.3] credentials:")
    print("  username=neo4j")
    print("  password=test12345")
    print("\n[Exercise 2.3] this run imports MySQL primary tables into Neo4j:")
    print("  - bankmarketing")
    print("  - livingwage50states")
    print("  (backup tables with suffix '_copy' are ignored)")

    compose_path = STEP_1_DIR / "compose.yaml"
    if compose_path.is_file():
        print("\n[Exercise 2.3] if services are not running, start them with:")
        print("  docker compose -f exercise_2/step_1/compose.yaml up -d")

    try:
        from neo4j import GraphDatabase
    except ImportError:
        print("\n[Exercise 2.3] neo4j driver not found.")
        print("Install with: python -m pip install -r exercise_2/requirements.txt")
        return

    try:
        from sqlalchemy import create_engine, text
    except ImportError:
        print("\n[Exercise 2.3] SQLAlchemy not found.")
        print("Install with: python -m pip install -r exercise_2/requirements.txt")
        return

    mysql_user = os.getenv("MYSQL_USER", "root").strip() or "root"
    mysql_password = os.getenv("MYSQL_PASSWORD", "pass")
    mysql_host = os.getenv("MYSQL_HOST", "localhost").strip() or "localhost"
    mysql_port = int(os.getenv("MYSQL_PORT", "3306").strip() or "3306")
    mysql_db = os.getenv("MYSQL_DATABASE", "test").strip() or "test"

    raw_limit = os.getenv("NEO4J_IMPORT_LIMIT", "5000").strip()
    try:
        import_limit = max(1, int(raw_limit))
    except ValueError:
        import_limit = 5000

    mysql_url = f"mysql+mysqlconnector://{mysql_user}:{mysql_password}@{mysql_host}:{mysql_port}/{mysql_db}"
    print(
        f"[Exercise 2.3] reading MySQL source: host={mysql_host} port={mysql_port} db={mysql_db} "
        "tables=bankmarketing,livingwage50states"
    )
    print(f"[Exercise 2.3] import limit: {import_limit} rows (env NEO4J_IMPORT_LIMIT)")

    try:
        engine = create_engine(mysql_url, echo=False)
        with engine.connect() as conn:
            mysql_tables = {row[0] for row in conn.execute(text("SHOW TABLES")).fetchall()}

        if "bankmarketing" not in mysql_tables:
            print("[Exercise 2.3] table 'bankmarketing' not found in MySQL.")
            print("Run first: python exercise_2/main.py --exercise 2_1")
            engine.dispose()
            return

        bank_df = pd.read_sql(
            text("SELECT * FROM bankmarketing LIMIT :limit"),
            engine,
            params={"limit": import_limit},
        )
        if "livingwage50states" in mysql_tables:
            living_df = pd.read_sql(
                text("SELECT * FROM livingwage50states LIMIT :limit"),
                engine,
                params={"limit": import_limit},
            )
        else:
            living_df = pd.DataFrame()
            print("[Exercise 2.3] warning: table 'livingwage50states' not found; continuing with bankmarketing only.")

        engine.dispose()
    except Exception as exc:
        print(f"[Exercise 2.3] failed to read MySQL source: {exc}")
        print("Verify MySQL is running and reachable from host.")
        return

    if bank_df.empty:
        print("[Exercise 2.3] no rows found in MySQL table 'bankmarketing'.")
        return

    def _norm_scalar(value):
        if pd.isna(value):
            return None
        if isinstance(value, str):
            value = value.strip()
            return value or None
        return value

    def _norm_label(value):
        value = _norm_scalar(value)
        if value is None:
            return None
        return str(value)

    bank_rows = []
    for idx, row in bank_df.reset_index(drop=True).iterrows():
        bank_rows.append(
            {
                "customer_id": f"bm_{idx + 1}",
                "age": None if pd.isna(row.get("age")) else int(row["age"]),
                "balance": None if pd.isna(row.get("balance")) else float(row["balance"]),
                "campaign": None if pd.isna(row.get("campaign")) else int(row["campaign"]),
                "pdays": None if pd.isna(row.get("pdays")) else int(row["pdays"]),
                "previous": None if pd.isna(row.get("previous")) else int(row["previous"]),
                "duration": None if pd.isna(row.get("duration")) else int(row["duration"]),
                "job": _norm_label(row.get("job")),
                "marital": _norm_label(row.get("marital")),
                "education": _norm_label(row.get("education")),
                "contact": _norm_label(row.get("contact")),
                "month": _norm_label(row.get("month")),
                "housing": _norm_label(row.get("housing")),
                "loan": _norm_label(row.get("loan")),
                "default_flag": _norm_label(row.get("default")),
                "poutcome": _norm_label(row.get("poutcome")),
                "outcome": _norm_label(row.get("y")),
            }
        )

    living_state_rows = []
    living_metric_rows = []
    if not living_df.empty and "state_territory" in living_df.columns:
        living_meta_cols = {"state_territory", "population_2020", "land_area_sqmi", "population_density"}
        living_metric_cols = [col for col in living_df.columns if col not in living_meta_cols]

        for _, row in living_df.iterrows():
            state_name = _norm_label(row.get("state_territory"))
            if not state_name:
                continue

            pop = _norm_scalar(row.get("population_2020"))
            land = _norm_scalar(row.get("land_area_sqmi"))
            density = _norm_scalar(row.get("population_density"))
            living_state_rows.append(
                {
                    "state": state_name,
                    "population_2020": None if pop is None else int(float(pop)),
                    "land_area_sqmi": None if land is None else float(land),
                    "population_density": None if density is None else float(density),
                }
            )

            for metric in living_metric_cols:
                wage = _norm_scalar(row.get(metric))
                if wage is None:
                    continue
                living_metric_rows.append(
                    {
                        "state": state_name,
                        "metric": metric,
                        "hourly_wage": float(wage),
                    }
                )

    uri = os.getenv("NEO4J_URI", "neo4j://localhost:7687").strip() or "neo4j://localhost:7687"
    user = os.getenv("NEO4J_USER", "neo4j").strip() or "neo4j"
    password = os.getenv("NEO4J_PASSWORD", "test12345")

    print(f"\n[Exercise 2.3] testing Python driver connection: uri={uri} user={user}")

    try:
        with GraphDatabase.driver(uri, auth=(user, password)) as driver:
            driver.verify_connectivity()
            print("[Exercise 2.3] connectivity check: OK")

            driver.execute_query("MATCH (c:BankCustomer) DETACH DELETE c")
            driver.execute_query("MATCH (s:LivingWageState) DETACH DELETE s")
            driver.execute_query("MATCH (h:HouseholdProfile) DETACH DELETE h")

            bank_import_query = """
            UNWIND $rows AS row
            MERGE (c:BankCustomer {id: row.customer_id})
            SET c.age = row.age,
                c.balance = row.balance,
                c.campaign = row.campaign,
                c.pdays = row.pdays,
                c.previous = row.previous,
                c.duration = row.duration

            FOREACH (_ IN CASE WHEN row.job IS NULL THEN [] ELSE [1] END |
              MERGE (j:Job {name: row.job})
              MERGE (c)-[:HAS_JOB]->(j)
            )
            FOREACH (_ IN CASE WHEN row.marital IS NULL THEN [] ELSE [1] END |
              MERGE (m:MaritalStatus {name: row.marital})
              MERGE (c)-[:HAS_MARITAL_STATUS]->(m)
            )
            FOREACH (_ IN CASE WHEN row.education IS NULL THEN [] ELSE [1] END |
              MERGE (e:EducationLevel {name: row.education})
              MERGE (c)-[:HAS_EDUCATION]->(e)
            )
            FOREACH (_ IN CASE WHEN row.contact IS NULL THEN [] ELSE [1] END |
              MERGE (ct:ContactChannel {name: row.contact})
              MERGE (c)-[:CONTACTED_BY]->(ct)
            )
            FOREACH (_ IN CASE WHEN row.month IS NULL THEN [] ELSE [1] END |
              MERGE (mo:CampaignMonth {name: row.month})
              MERGE (c)-[:CONTACTED_IN_MONTH]->(mo)
            )
            FOREACH (_ IN CASE WHEN row.housing IS NULL THEN [] ELSE [1] END |
              MERGE (h:HousingLoanFlag {name: row.housing})
              MERGE (c)-[:HAS_HOUSING_LOAN]->(h)
            )
            FOREACH (_ IN CASE WHEN row.loan IS NULL THEN [] ELSE [1] END |
              MERGE (l:PersonalLoanFlag {name: row.loan})
              MERGE (c)-[:HAS_PERSONAL_LOAN]->(l)
            )
            FOREACH (_ IN CASE WHEN row.default_flag IS NULL THEN [] ELSE [1] END |
              MERGE (d:DefaultFlag {name: row.default_flag})
              MERGE (c)-[:IN_DEFAULT]->(d)
            )
            FOREACH (_ IN CASE WHEN row.poutcome IS NULL THEN [] ELSE [1] END |
              MERGE (po:PreviousOutcome {name: row.poutcome})
              MERGE (c)-[:HAS_PREVIOUS_OUTCOME]->(po)
            )
            FOREACH (_ IN CASE WHEN row.outcome IS NULL THEN [] ELSE [1] END |
              MERGE (o:CampaignOutcome {name: row.outcome})
              MERGE (c)-[:HAS_OUTCOME]->(o)
            )
            """
            driver.execute_query(bank_import_query, rows=bank_rows)
            print(f"[Exercise 2.3] imported {len(bank_rows)} bankmarketing rows into Neo4j")

            if living_state_rows:
                state_import_query = """
                UNWIND $rows AS row
                MERGE (s:LivingWageState {name: row.state})
                SET s.population_2020 = row.population_2020,
                    s.land_area_sqmi = row.land_area_sqmi,
                    s.population_density = row.population_density
                """
                living_metric_query = """
                UNWIND $rows AS row
                MATCH (s:LivingWageState {name: row.state})
                MERGE (h:HouseholdProfile {name: row.metric})
                MERGE (s)-[r:HAS_LIVING_WAGE]->(h)
                SET r.hourly_wage = row.hourly_wage
                """
                driver.execute_query(state_import_query, rows=living_state_rows)
                driver.execute_query(living_metric_query, rows=living_metric_rows)
                print(
                    "[Exercise 2.3] imported "
                    f"{len(living_state_rows)} livingwage state rows and {len(living_metric_rows)} wage metrics into Neo4j"
                )
            else:
                print("[Exercise 2.3] no livingwage50states rows imported.")

            print("\n[Exercise 2.3] import completed.")
            print("[Exercise 2.3] analytical Cypher queries are no longer executed automatically.")
            print("[Exercise 2.3] open Neo4j Browser and run the sample queries listed in the README.")
    except Exception as exc:
        print(f"[Exercise 2.3] Neo4j connection/query failed: {exc}")
        print("Verify neo4j service is up and credentials are correct.")


def exercise_2_4() -> None:
    """Import MySQL primary tables into OpenSearch and guide Dashboards usage."""
    dashboards_url = "http://localhost:5601"
    opensearch_url = "http://localhost:9200"
    username = os.getenv("OPENSEARCH_USER", "admin").strip() or "admin"
    password = os.getenv("OPENSEARCH_PASSWORD", "@StrongP4ssword!")
    bank_index = os.getenv("OPENSEARCH_BANK_INDEX", os.getenv("OPENSEARCH_INDEX", "bankmarketing")).strip() or "bankmarketing"
    living_index = os.getenv("OPENSEARCH_LIVINGWAGE_INDEX", "livingwage50states").strip() or "livingwage50states"
    raw_limit = os.getenv("OPENSEARCH_IMPORT_LIMIT", "5000").strip()
    try:
        import_limit = max(1, int(raw_limit))
    except ValueError:
        import_limit = 5000

    print(f"[Exercise 2.4] OpenSearch Dashboards URL: {dashboards_url}")
    print("[Exercise 2.4] credentials:")
    print(f"  user={username}")
    print(f"  password={password}")
    print("\n[Exercise 2.4] this run imports MySQL primary tables into OpenSearch:")
    print(f"  - bankmarketing -> index '{bank_index}'")
    print(f"  - livingwage50states -> index '{living_index}'")
    print("  (backup tables with suffix '_copy' are ignored)")
    print(f"[Exercise 2.4] import limit: {import_limit} rows per table")

    compose_path = STEP_1_DIR / "compose.yaml"
    if compose_path.is_file():
        print("\n[Exercise 2.4] if services are not running, start them with:")
        print("  docker compose -f exercise_2/step_1/compose.yaml up -d")
    else:
        print("\n[Exercise 2.4] warning: compose file not found at exercise_2/step_1/compose.yaml")

    try:
        with urllib.request.urlopen(dashboards_url, timeout=5) as resp:
            print(f"\n[Exercise 2.4] Dashboards reachable: HTTP {resp.status}")
    except Exception as exc:
        print(f"\n[Exercise 2.4] Dashboards check failed: {exc}")
        print("Start the stack, then open http://localhost:5601 in your browser.")
        return

    try:
        from sqlalchemy import create_engine, text
    except ImportError:
        print("[Exercise 2.4] SQLAlchemy is not installed.")
        print("Install with: python -m pip install -r exercise_2/requirements.txt")
        return

    mysql_user = os.getenv("MYSQL_USER", "root").strip() or "root"
    mysql_password = os.getenv("MYSQL_PASSWORD", "pass")
    mysql_host = os.getenv("MYSQL_HOST", "localhost").strip() or "localhost"
    mysql_port = int(os.getenv("MYSQL_PORT", "3306").strip() or "3306")
    mysql_db = os.getenv("MYSQL_DATABASE", "test").strip() or "test"
    mysql_url = f"mysql+mysqlconnector://{mysql_user}:{mysql_password}@{mysql_host}:{mysql_port}/{mysql_db}"

    print(
        f"[Exercise 2.4] reading MySQL source: host={mysql_host} port={mysql_port} db={mysql_db} "
        "tables=bankmarketing,livingwage50states"
    )
    try:
        engine = create_engine(mysql_url, echo=False)
        with engine.connect() as conn:
            mysql_tables = {row[0] for row in conn.execute(text("SHOW TABLES")).fetchall()}

        if "bankmarketing" not in mysql_tables:
            print("[Exercise 2.4] table 'bankmarketing' not found in MySQL.")
            print("Run first: python exercise_2/main.py --exercise 2_1")
            engine.dispose()
            return

        bank_df = pd.read_sql(
            text("SELECT * FROM bankmarketing LIMIT :limit"),
            engine,
            params={"limit": import_limit},
        )
        if "livingwage50states" in mysql_tables:
            living_df = pd.read_sql(
                text("SELECT * FROM livingwage50states LIMIT :limit"),
                engine,
                params={"limit": import_limit},
            )
        else:
            living_df = pd.DataFrame()
            print("[Exercise 2.4] warning: table 'livingwage50states' not found; continuing with bankmarketing only.")

        engine.dispose()
    except Exception as exc:
        print(f"[Exercise 2.4] failed to read MySQL source: {exc}")
        return

    if bank_df.empty:
        print("[Exercise 2.4] no rows found in MySQL table 'bankmarketing'.")
        return

    def _to_json_serializable(value):
        if pd.isna(value):
            return None
        if isinstance(value, (np.integer,)):
            return int(value)
        if isinstance(value, (np.floating,)):
            return float(value)
        if isinstance(value, (np.bool_,)):
            return bool(value)
        return value

    def _opensearch_request(path: str, method: str = "GET", payload=None, content_type: str = "application/json"):
        url = f"{opensearch_url.rstrip('/')}/{path.lstrip('/')}"
        data = None
        if payload is not None:
            if isinstance(payload, bytes):
                data = payload
            elif isinstance(payload, str):
                data = payload.encode("utf-8")
            else:
                data = json.dumps(payload).encode("utf-8")

        request = urllib.request.Request(url, method=method, data=data)
        request.add_header("Content-Type", content_type)
        if username:
            auth_raw = f"{username}:{password}".encode("utf-8")
            auth_header = b64encode(auth_raw).decode("ascii")
            request.add_header("Authorization", f"Basic {auth_header}")

        with urllib.request.urlopen(request, timeout=15) as resp:
            body = resp.read().decode("utf-8") if resp.readable() else ""
            return resp.status, body

    def _recreate_index(index_name: str):
        try:
            _opensearch_request(index_name, "DELETE")
        except urllib.error.HTTPError:
            pass
        _opensearch_request(index_name, "PUT", payload={})

    def _bulk_index_dataframe(df: pd.DataFrame, index_name: str, source_table: str, id_prefix: str):
        _recreate_index(index_name)

        lines = []
        for i, rec in enumerate(df.to_dict(orient="records"), start=1):
            doc = {k: _to_json_serializable(v) for k, v in rec.items()}
            doc["source_table"] = source_table
            lines.append(json.dumps({"index": {"_index": index_name, "_id": f"{id_prefix}_{i}"}}))
            lines.append(json.dumps(doc))
        bulk_payload = "\n".join(lines) + "\n"

        _, bulk_body = _opensearch_request(
            "_bulk?refresh=true",
            "POST",
            payload=bulk_payload,
            content_type="application/x-ndjson",
        )
        bulk_json = json.loads(bulk_body) if bulk_body else {}
        return bool(bulk_json.get("errors")), len(df)

    try:
        status, _ = _opensearch_request("/", "GET")
        print(f"[Exercise 2.4] OpenSearch API reachable: HTTP {status}")
    except urllib.error.HTTPError as exc:
        print(f"[Exercise 2.4] OpenSearch API check returned HTTP {exc.code}")
        print("Check OpenSearch credentials/security settings in compose.")
        return
    except Exception as exc:
        print(f"[Exercise 2.4] OpenSearch API check failed: {exc}")
        print("Verify the OpenSearch container is running and port 9200 is exposed.")
        return

    try:
        bank_errors, bank_count = _bulk_index_dataframe(bank_df, bank_index, "bankmarketing", "bm")
        if bank_errors:
            print(f"[Exercise 2.4] bankmarketing import completed with errors for index '{bank_index}'.")
        else:
            print(f"[Exercise 2.4] imported {bank_count} rows into OpenSearch index '{bank_index}'")

        if not living_df.empty:
            living_errors, living_count = _bulk_index_dataframe(
                living_df,
                living_index,
                "livingwage50states",
                "lw",
            )
            if living_errors:
                print(f"[Exercise 2.4] livingwage import completed with errors for index '{living_index}'.")
            else:
                print(f"[Exercise 2.4] imported {living_count} rows into OpenSearch index '{living_index}'")
        else:
            print("[Exercise 2.4] no livingwage50states rows imported.")

        print("\n[Exercise 2.4] import completed.")
        print("[Exercise 2.4] OpenSearch analytical queries are no longer executed automatically.")
        print("[Exercise 2.4] open Dashboards Dev Tools and run the sample queries listed in the README.")
    except Exception as exc:
        print(f"[Exercise 2.4] import/search pipeline failed: {exc}")
        print("Verify OpenSearch version compatibility and security options.")
        return

    print("\n[Exercise 2.4] next steps in Dashboards:")
    print("1. Open http://localhost:5601.")
    print("2. Log in with admin / @StrongP4ssword! (if prompted).")
    print(f"3. Create Data Views for indexes: {bank_index} and {living_index}.")
    print("4. Open Dev Tools -> Console and run the sample queries in the README.")
    print("5. Open Discover and inspect documents from both tables.")
    print("6. Build dashboards for bank outcome and living wage comparisons.")


EXERCISE_HANDLERS = {
    "2_1": exercise_2_1,
    "2_2": exercise_2_2,
    "2_3": exercise_2_3,
    "2_4": exercise_2_4,
}


def run_exercises(exercise_ids: list[str]) -> None:
    """Run a list of Exercise 2 steps in order."""
    for exercise_id in exercise_ids:
        print(f"\n[Runner] running exercise {exercise_id}")
        EXERCISE_HANDLERS[exercise_id]()
