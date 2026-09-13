"""Celery application factory.

Workers are only needed in ``WORKER_MODE=async``; in the default ``sync`` mode
the ingestion pipeline runs in-process inside the FastAPI bridge, so a Redis
instance is *not* required for local development.
"""

from __future__ import annotations

from celery import Celery

from .config import get_settings


def make_celery() -> Celery:
    settings = get_settings()
    app = Celery(
        "chronos",
        broker=settings.celery_broker_url,
        backend=settings.celery_result_backend,
        include=["chronos.tasks.ingest"],
    )
    app.conf.update(
        task_serializer="json",
        accept_content=["json"],
        result_serializer="json",
        timezone="UTC",
        enable_utc=True,
        task_track_started=True,
        task_time_limit=60 * 30,  # 30 minutes per ingestion job
        worker_max_tasks_per_child=20,
        task_acks_late=True,
    )
    return app


celery_app = make_celery()