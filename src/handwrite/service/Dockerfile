# Multi-stage build for the HandWrite FastAPI service.

# ---------- builder ---------- #
FROM python:3.11-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /build

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        git \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml ./pyproject.toml
COPY src ./src

RUN pip install --upgrade pip \
    && pip wheel --wheel-dir /wheels . \
    && pip wheel --wheel-dir /wheels fastapi "uvicorn[standard]"

# ---------- runtime ---------- #
FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HANDWRITE_SERVICE_HOST=0.0.0.0 \
    HANDWRITE_SERVICE_PORT=8000

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        tesseract-ocr \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --shell /bin/bash handwrite

COPY --from=builder /wheels /wheels

RUN pip install --upgrade pip \
    && pip install --no-index --find-links=/wheels handwrite fastapi "uvicorn[standard]" \
    && rm -rf /wheels

USER handwrite

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "handwrite.service.app:create_app", \
     "--factory", "--host", "0.0.0.0", "--port", "8000"]
