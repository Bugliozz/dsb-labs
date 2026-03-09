# Exercise 1.7 - Flask + Docker

This folder replicates the Flask + Docker workflow from class.

You can validate that the committed scaffold is present with:

```bash
python exercise_1/main.py --exercise 1_7
```

## 1. Run locally (without Docker)

```bash
python -m pip install -r exercise_1/requirements.txt
python app.py
```

Open:
`http://localhost:5000/`

## 2. Build Docker image

```bash
docker build --tag python-docker .
```

## 3. Run container

```bash
docker run -d -p 5000:5000 --name python-docker-lab python-docker
docker ps
```

Open:
`http://localhost:5000/`

## 4. Stop and remove container

```bash
docker stop python-docker-lab
docker rm python-docker-lab
```

## 5. Optional cleanup

```bash
docker image rm python-docker
```
