FROM python:3.14-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV FLIGHTS_OUTPUT_DIR=/python-docker/malpensa_flights

WORKDIR /python-docker

COPY requirements.txt app.py ./

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

RUN mkdir -p /python-docker/malpensa_flights

CMD ["python", "app.py"]
