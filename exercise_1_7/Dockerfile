FROM python:3.8-slim-buster
WORKDIR /python-docker
COPY requirements.txt app.py ./
RUN pip install --upgrade pip
RUN pip install -r requirements.txt
CMD ["python3", "app.py"]
