# syntax=docker/dockerfile:1
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src
COPY main.py ./

RUN python -m pip install --upgrade pip \
    && python -m pip install .

RUN mkdir -p /app/artifacts

ENTRYPOINT ["breast-cancer-baseline"]
CMD ["--help"]
