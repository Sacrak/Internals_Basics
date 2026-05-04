FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src ./src
COPY models ./models

ENV PYTHONPATH=/app/src

# Default container command runs the JSON-emitting CLI predictor.
ENTRYPOINT ["python", "src/predict_cli.py"]
