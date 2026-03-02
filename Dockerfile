FROM python:3.11-slim

WORKDIR /app

COPY fern-health/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY fern-health/ .

CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}"]
