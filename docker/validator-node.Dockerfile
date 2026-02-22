FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc \
        g++ \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY docker/validator-runtime-requirements.txt /tmp/validator-runtime-requirements.txt

RUN python -m pip install --upgrade pip \
    && python -m pip install --no-cache-dir torch==2.9.1 --index-url https://download.pytorch.org/whl/cpu \
    && python -m pip install --no-cache-dir -r /tmp/validator-runtime-requirements.txt

COPY . /app

RUN chmod +x /app/docker/run-validator.sh

ENV PYTHONPATH=/app

ENTRYPOINT ["/app/docker/run-validator.sh"]
