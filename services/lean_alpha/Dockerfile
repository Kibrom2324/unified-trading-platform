FROM python:3.11-slim
WORKDIR /app
RUN apt-get update && apt-get install -y build-essential librdkafka-dev && rm -rf /var/lib/apt/lists/*
COPY shared/ ./shared/
COPY services/lean_alpha/ ./services/lean_alpha/
COPY configs/ ./configs/
RUN pip install --no-cache-dir confluent-kafka structlog asyncpg python-dotenv pyyaml
ENV PYTHONPATH=/app
CMD ["python", "-m", "services.lean_alpha.main"]
