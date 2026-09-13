"""End-to-end demo seed: generate PDFs, upload + ingest, then run sample
temporal questions against the running stack.

Prerequisites: the worker bridge (FastAPI :8100) and the Express API (:4000)
must be running (see README Quickstart / Docker).

Usage:
    python scripts/seed_demo.py [--api-url http://localhost:4000]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import httpx

from make_sample_pdfs import build_cross_border_policy, build_tax_code_amendments

SAMPLE_QUESTIONS = [
    {
        "question": "What was the compliance protocol for cross-border data transfers in Q3 2024?",
        "window": "Q3 2024",
    },
    {
        "question": "What was the cross-border data transfer policy in Q1 2023 versus the current regime?",
        "window": "Q1 2023",
    },
    {
        "question": "What is the standard corporate income tax rate for 2025?",
        "window": "2025",
    },
    {
        "question": "Which entities are exempt from the corporate income tax and until when?",
        "window": "",
    },
]

COMPARE_QUESTIONS = [
    {
        "question": "How did the cross-border data transfer compliance obligations change between 2023 and the current regime?",
        "window_a": "2023",
        "window_b": "Q4 2024",
    },
    {
        "question": "What changed in the corporate tax code between 2023 and 2025?",
        "window_a": "2023",
        "window_b": "2025",
    },
]


def wait_until_indexed(client: httpx.Client, api: str, doc_id: str, timeout: float = 120.0) -> dict:
    deadline = time.time() + timeout
    while time.time() < deadline:
        rec = client.get(f"{api}/api/documents/{doc_id}").json()
        status = rec.get("status", "")
        if status in ("indexed", "failed"):
            return rec
        time.sleep(1.0)
    raise TimeoutError(f"document {doc_id} did not finish ingesting in {timeout}s")


def main() -> None:
    parser = argparse.ArgumentParser(description="Seed ChronosAudit with sample documents and queries")
    parser.add_argument("--api-url", default="http://localhost:4000", help="Express API base URL")
    parser.add_argument("--sample-dir", default="sample", help="where to write/generate PDFs")
    parser.add_argument("--skip-queries", action="store_true", help="ingest only, skip the sample Q&A")
    args = parser.parse_args()

    api = args.api_url.rstrip("/")
    sample_dir = Path(args.sample_dir)
    sample_dir.mkdir(parents=True, exist_ok=True)

    pdf1 = sample_dir / "cross_border_data_policy.pdf"
    pdf2 = sample_dir / "tax_code_amendments_2023_2025.pdf"
    if not pdf1.exists():
        build_cross_border_policy(pdf1)
    if not pdf2.exists():
        build_tax_code_amendments(pdf2)

    print(f"→ Seeding against {api}")
    with httpx.Client(timeout=60) as client:
        # -- health -----------------------------------------------------------
        try:
            health = client.get(f"{api}/api/system/health").json()
            print(f"  system health: {health.get('service')} | model={health.get('model_provider')}")
        except httpx.HTTPError as exc:
            print(f"  ✗ Cannot reach the API: {exc}", file=sys.stderr)
            sys.exit(1)

        # -- ingest -----------------------------------------------------------
        doc_ids = []
        for pdf in (pdf1, pdf2):
            print(f"→ Uploading {pdf.name} …")
            with pdf.open("rb") as fh:
                resp = client.post(
                    f"{api}/api/documents/upload",
                    files={"file": (pdf.name, fh, "application/pdf")},
                    data={"wait": "true", "timeout_seconds": "180"},
                )
            resp.raise_for_status()
            record = resp.json()
            doc_ids.append(record["id"])
            state = record.get("status")
            print(f"  ✓ {record['id']} status={state} chunks={record.get('chunk_count')} "
                  f"entities={record.get('entity_count')} relations={record.get('relation_count')}")

        # -- graph stats ------------------------------------------------------
        stats = client.get(f"{api}/api/graph/stats").json()
        print(f"→ Graph now has {stats['node_count']} nodes, {stats['edge_count']} edges\n")

        if args.skip_queries:
            return

        # -- sample Q&A -------------------------------------------------------
        for q in SAMPLE_QUESTIONS:
            print(f"Q: {q['question']}")
            payload = {"question": q["question"]}
            if q.get("window"):
                payload["window"] = q["window"]
            resp = client.post(f"{api}/api/query", json=payload)
            resp.raise_for_status()
            data = resp.json()
            answer = data.get("answer", "")
            print("A:", answer[:400].replace("\n", "\n   "), "\n")

        # -- sample comparisons ------------------------------------------------
        for q in COMPARE_QUESTIONS:
            print(f"⇄ {q['question']}")
            resp = client.post(
                f"{api}/api/query/compare",
                json={"question": q["question"], "window_a": q["window_a"], "window_b": q["window_b"]},
            )
            resp.raise_for_status()
            data = resp.json()
            print(data.get("narrative", "")[:600].replace("\n", "\n   "), "\n")


if __name__ == "__main__":
    main()