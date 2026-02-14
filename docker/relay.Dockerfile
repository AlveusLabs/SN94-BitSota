FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
    && rm -rf /var/lib/apt/lists/*

ARG BITSOTA_GIT_REF=main

WORKDIR /opt/bitsota

RUN git clone --depth 1 --branch "${BITSOTA_GIT_REF}" https://github.com/AlveusLabs/BitSota.git /opt/bitsota

RUN python -m pip install --upgrade pip \
    && python -m pip install --no-cache-dir -r relay/requirements.txt

ENV PYTHONPATH=/opt/bitsota

ENTRYPOINT ["python", "-m", "relay"]
