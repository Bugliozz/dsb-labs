# Exercise 4 - Malpensa flight collector

This folder contains the Flask service used to scrape daily arrivals and departures from `milanomalpensa-airport.com` and persist the JSON output either:

- locally, under `exercise_4/malpensa_flights/`, for repo backup and local testing;
- in Google Cloud Storage, when `BUCKET_NAME` is configured for Cloud Run.

## Local usage

### Setup

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r exercise_4/requirements.txt
```

### Start the local Flask app

```bash
python exercise_4/main.py
```

Open:

- `http://localhost:8080/health`
- `http://localhost:8080/?date=2026-03-14`

When `BUCKET_NAME` is not set, the generated files are written to `exercise_4/malpensa_flights/`.

### Run one collection cycle without the web server

```bash
python exercise_4/main.py collect --date 2026-03-14
```

## Docker

Inside the container, local backups are written to `/python-docker/malpensa_flights`.

```bash
docker build --tag milanomalpensa exercise_4
docker run --rm -p 8080:8080 -v ${PWD}/exercise_4/mxp_flights:/python-docker/malpensa_flights milanomalpensa
```

## Google Cloud deployment

The currently verified GCP setup, checked on **2026-03-16**, is:

- project: `linear-potion-490413-s9`
- region: `europe-west1`
- Cloud Run service: `milanomalpensa`
- bucket: `malpensa-flights-marco`
- scheduler job: `malpensa-flights-job`

### Create the storage bucket

```bash
gcloud config set project linear-potion-490413-s9
gcloud storage buckets create gs://malpensa-flights-marco --default-storage-class=STANDARD --location=EUROPE-WEST1
```

### Deploy to Cloud Run

The deployed service uses `BUCKET_NAME` and uploads through the Storage API, so Cloud Storage volume mounts are not required.

```bash
gcloud run deploy milanomalpensa --source exercise_4 --region=europe-west1 --max-instances=1 --allow-unauthenticated --set-env-vars BUCKET_NAME=malpensa-flights-marco
```

### Create the scheduler job

```bash
gcloud scheduler jobs create http malpensa-flights-job --schedule "0 11 * * *" --uri "https://milanomalpensa-47880508774.europe-west1.run.app/" --location=europe-west1 --http-method GET
```

## Verified runtime checks

The following checks passed on **2026-03-16**:

- Cloud Run `milanomalpensa` was `Ready`
- direct HTTP call to `/?date=2026-03-14` returned success
- bucket uploads were visible for `2026-03-14_A.json` and `2026-03-14_D.json`
- a manual `gcloud scheduler jobs run malpensa-flights-job --location=europe-west1` updated the `2026-03-15_*.json` objects
