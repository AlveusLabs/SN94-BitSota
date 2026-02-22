FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libgomp1 \
        patchelf \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

COPY requirements.txt /workspace/requirements.txt

RUN python -m pip install --upgrade pip \
    && python -m pip install --no-cache-dir pyinstaller \
    && python -m pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu \
    && python -m pip install --no-cache-dir -r /workspace/requirements.txt

COPY . /workspace

ENV PYTHONPATH=/workspace

CMD ["bash", "-lc", "pyinstaller bitsota.spec && ls -lah dist"]
