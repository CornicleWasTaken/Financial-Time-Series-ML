# syntax=docker/dockerfile:1.7
#
# Multi-stage Dockerfile for the financial-ml skeleton.
#
# Stage 1 ("builder") installs build tooling and the project's runtime
# dependencies into a virtualenv. Stage 2 ("runtime") copies that venv and
# the application code into a slim image that exposes the FastAPI service
# (Phase 11) on port 8000.

ARG PYTHON_VERSION=3.12

# ---------------------------------------------------------------------------
# Stage 1 — builder
# ---------------------------------------------------------------------------
FROM python:${PYTHON_VERSION}-slim AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /build

# System packages required to build Python wheels (kept minimal).
RUN apt-get update \
    && apt-get install --no-install-recommends -y build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml ./
COPY src ./src

# Install the package + its runtime + dev dependencies into a venv we
# copy wholesale into the runtime stage.
RUN python -m venv /opt/venv \
    && /opt/venv/bin/pip install --upgrade pip \
    && /opt/venv/bin/pip install ".[dev]"

# ---------------------------------------------------------------------------
# Stage 2 — runtime
# ---------------------------------------------------------------------------
FROM python:${PYTHON_VERSION}-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:${PATH}" \
    PYTHONPATH="/app/src"

# Non-root user for runtime safety.
RUN groupadd --system app && useradd --system --gid app --home /app app

WORKDIR /app

COPY --from=builder /opt/venv /opt/venv
COPY --from=builder /build/pyproject.toml ./pyproject.toml
COPY src ./src
COPY configs ./configs

# Project data is expected to be bind-mounted; see docker-compose.yml.
RUN mkdir -p /app/data/raw /app/data/processed /app/data/features \
             /app/models /app/mlruns \
    && chown -R app:app /app

USER app

EXPOSE 8000

# Health check uses the liveness endpoint defined in Phase 11. Until that
# lands, the command just prints a banner so `docker compose run app` is
# obviously a no-op.
CMD ["python", "-c", "from financial_ml import __version__ as v; print(f'financial-ml {v} (Phase 0 skeleton)')"]
