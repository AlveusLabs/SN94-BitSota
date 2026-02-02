FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc \
        g++ \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY relay/requirements.txt /tmp/relay-requirements.txt
RUN pip install --no-cache-dir -r /tmp/relay-requirements.txt \
    && pip install --no-cache-dir \
        numpy==2.3.5 \
        pandas==2.3.3 \
        scikit-learn==1.7.2

COPY core/ ./core/
COPY validator/ ./validator/
COPY validator_hyperparams.json ./validator_hyperparams.json
COPY miner_hyperparams.json ./miner_hyperparams.json
