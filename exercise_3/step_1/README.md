# Exercise 3.1 - Ames House Price API

This step implements the course exercise about estimating house prices in Ames, Iowa with a dockerized API service.

## Dataset placement

Place these files inside `exercise_3/step_1/data/`:

- `train.csv`
- `data_description.txt`

The training workflow expects the Kaggle Ames `train.csv` file and uses `SalePrice` as the regression target.

## 1. Train offline

From the project root:

```bash
python main.py --exercise 3_1 --command train
python exercise_3/main.py train
```

What it does:

- validates the local dataset files;
- creates EDA plots under `exercise_3/step_1/images/`;
- trains and compares multiple regressors with 5-fold cross validation;
- saves the best pipeline and service metadata under `exercise_3/step_1/artifacts/`.

## 2. Run locally

```bash
python main.py --exercise 3_1 --command serve
python exercise_3/main.py serve
```

Open:
`http://localhost:5000/`

Health check:

```bash
curl http://localhost:5000/api/health
```

Demo prediction:

```bash
curl "http://localhost:5000/api?data=[10603,1977,1610,2,68]"
```

Advanced prediction:

```bash
curl -X POST http://localhost:5000/api/predict \
  -H "Content-Type: application/json" \
  -d "{\"LotArea\":10603,\"YearBuilt\":1977,\"GrLivArea\":1610,\"GarageCars\":2,\"OpenPorchSF\":68}"
```

## 3. Build Docker image

```bash
docker build -t ames-house-price-api exercise_3/step_1
```

## 4. Run Docker container

```bash
docker run --rm -p 5000:5000 ames-house-price-api
```

The Docker image contains only the inference service and prebuilt artifacts. Train the model first from the repo root so `exercise_3/step_1/artifacts/` exists before building the image.

## 5. Deploy to Cloud Run

If the lab for **2026-03-16** only requires publishing the already completed API, this step is the missing operational piece.

From the project root:

```bash
gcloud config set project linear-potion-490413-s9
gcloud run deploy ames-house-price-api --source exercise_3/step_1 --region=europe-west1 --allow-unauthenticated --port 5000
```

After deployment, verify:

```bash
curl https://YOUR-SERVICE-URL/api/health
curl "https://YOUR-SERVICE-URL/api?data=[10603,1977,1610,2,68]"
```

Verified deployment on `2026-03-16`:

- service: `ames-house-price-api`
- URL: `https://ames-house-price-api-47880508774.europe-west1.run.app`
