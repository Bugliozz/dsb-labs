FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

RUN mkdir -p /app/exercise_3/step_1 \
    && printf '"""Exercise 3 package."""\n' > /app/exercise_3/__init__.py

COPY __init__.py /app/exercise_3/step_1/__init__.py
COPY app.py ames.py /app/exercise_3/step_1/
COPY templates /app/exercise_3/step_1/templates
COPY artifacts /app/exercise_3/step_1/artifacts

EXPOSE 5000

CMD ["python", "-m", "exercise_3.step_1.app"]
