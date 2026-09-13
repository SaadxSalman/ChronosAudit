FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY chronos ./chronos
COPY tests ./tests

RUN mkdir -p /app/data/{uploads,state,graph/exports,lancedb}

EXPOSE 8100

CMD ["uvicorn", "chronos.api:app", "--host", "0.0.0.0", "--port", "8100"]