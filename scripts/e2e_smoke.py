#!/usr/bin/env python3
"""Full-stack end-to-end smoke test for ChronosAudit.

Boots the real worker bridge (FastAPI/uvicorn) and the real Express API on
ephemeral ports with throwaway data directories, then drives the exact flow an
auditor would:

  1. GET  /api/health                     -> both services up, worker upstream ok
  2. POST /api/documents/upload  (x2)     -> sample regulatory PDFs ingest sync
  3. POST /api/query             (x2)     -> windowed answers with evidence
  4. POST /api/query/compare              -> added/removed facts between regimes
  5. GET  /api/graph/stats                -> non-trivial temporal graph

Exit code 0 = smoke passed. Requires: Python deps installed, `node` on PATH,
and the API built (`npm run build` in services/api) — rebuilt automatically
when missing.

Usage:
    python scripts/e2e_smoke.py [--keep]        # --keep retains artifacts dir
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_sample_pdfs import build_cross_border_policy, build_tax_code_amendments  # noqa: E402

WORKER_DIR = ROOT / "services" / "worker"
API_DIR = ROOT / "services" / "api"
WEB_DIR = ROOT / "web"


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def http_json(method: str, url: str, body: dict | None = None, timeout: float = 30.0):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method=method,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as res:
        return json.loads(res.read().decode())


def http_multipart(url: str, field: str, filename: str, content: bytes,
                   extra: dict[str, str], timeout: float = 240.0):
    boundary = uuid.uuid4().hex
    parts = []
    for key, value in extra.items():
        parts.append(
            f"--{boundary}\r\nContent-Disposition: form-data; name=\"{key}\"\r\n\r\n{value}\r\n".encode()
        )
    parts.append(
        f"--{boundary}\r\nContent-Disposition: form-data; name=\"{field}\"; "
        f"filename=\"{filename}\"\r\nContent-Type: application/pdf\r\n\r\n".encode()
        + content + b"\r\n"
    )
    parts.append(f"--{boundary}--\r\n".encode())
    req = urllib.request.Request(
        url, data=b"".join(parts), method="POST",
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as res:
        return json.loads(res.read().decode())


def wait_health(url: str, proc: subprocess.Popen, name: str, deadline: float = 60.0) -> dict:
    end = time.time() + deadline
    last_err = "never polled"
    while time.time() < end:
        if proc.poll() is not None:
            raise RuntimeError(f"{name} exited early with code {proc.returncode}")
        try:
            with urllib.request.urlopen(url, timeout=3) as res:
                return json.loads(res.read().decode())
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            last_err = str(exc)
            time.sleep(0.5)
    raise RuntimeError(f"{name} did not become healthy at {url}: {last_err}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--keep", action="store_true", help="keep artifacts (data dir, PDFs) for inspection")
    args = ap.parse_args()

    if not (API_DIR / "dist" / "index.js").exists():
        print("→ dist/index.js missing; building API…")
        subprocess.run("npm run build", cwd=API_DIR, shell=True, check=True)

    artifacts = Path(tempfile.mkdtemp(prefix="chronos-e2e-"))
    wport, aport = free_port(), free_port()
    worker_base = f"http://127.0.0.1:{wport}"
    api_base = f"http://127.0.0.1:{aport}"

    base_env = {
        **os.environ,
        "PYTHONIOENCODING": "utf-8",
        "WORKER_MODE": "sync",
        "LANCE_DB_PATH": str(artifacts / "lancedb"),
        "GRAPH_STORE_PATH": str(artifacts / "graph" / "chronos.gpickle"),
        "GRAPH_EXPORT_PATH": str(artifacts / "graph" / "exports"),
        "DOC_STATE_DIR": str(artifacts / "state"),
        "UPLOAD_DIR": str(artifacts / "uploads"),
        "MODEL_PROVIDER": "none",
        "EMBEDDING_PROVIDER": "hashing",
    }
    api_env = {
        **os.environ,
        "PORT": str(aport),
        "WORKER_BASE_URL": worker_base,
        "DATA_DIR": str(artifacts / "api"),
        "WEB_DIR": str(WEB_DIR),
    }

    procs: list[subprocess.Popen] = []
    try:
        print(f"→ booting worker bridge on :{wport} (sync mode, artifacts in {artifacts})")
        procs.append(subprocess.Popen(
            [sys.executable, "-m", "uvicorn", "chronos.api:app",
             "--host", "127.0.0.1", "--port", str(wport), "--log-level", "warning"],
            cwd=WORKER_DIR, env=base_env,
            stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT,
        ))
        print(f"→ booting express api on :{aport}")
        procs.append(subprocess.Popen(
            ["node", "dist/index.js"], cwd=API_DIR, env=api_env,
            stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT,
        ))

        wh = wait_health(f"{worker_base}/health", procs[0], "worker")
        assert wh.get("ok"), f"worker health not ok: {wh}"
        ah = wait_health(f"{api_base}/api/health", procs[1], "api")
        assert ah.get("ok") and ah.get("worker_upstream"), f"api health not ok: {ah}"
        print(f"  ✓ health ok — worker mode={wh.get('mode')}, embeddings={wh.get('embedding_provider')}")

        # -- ingest ---------------------------------------------------------
        pdfs = artifacts / "pdfs"
        pdfs.mkdir()
        p1 = pdfs / "cross_border_data_policy.pdf"
        p2 = pdfs / "tax_code_amendments_2023_2025.pdf"
        build_cross_border_policy(p1)
        build_tax_code_amendments(p2)
        for path in (p1, p2):
            rec = http_multipart(f"{api_base}/api/documents/upload", "file",
                                 path.name, path.read_bytes(),
                                 {"wait": "true", "timeout_seconds": "180"})
            assert rec.get("status") == "indexed", f"{path.name} not indexed: {rec}"
            assert (rec.get("chunk_count") or 0) >= 1, f"no chunks for {path.name}: {rec}"
            print(f"  ✓ indexed {path.name}: chunks={rec['chunk_count']} "
                  f"entities={rec.get('entity_count')} relations={rec.get('relation_count')}")

        # -- windowed questions ---------------------------------------------
        r1 = http_json("POST", f"{api_base}/api/query",
                       {"question": "What was the compliance protocol for cross-border data transfers?",
                        "window": "Q3 2024"})
        assert r1.get("answer"), "empty answer for Q3 2024"
        assert r1["context"]["hits"], "no vector hits for Q3 2024"
        print(f"  ✓ Q3 2024 answer: {str(r1['answer'])[:90]}…")

        r2 = http_json("POST", f"{api_base}/api/query",
                       {"question": "What is the standard corporate income tax rate?",
                        "window": "2025"})
        assert r2.get("answer"), "empty answer for 2025"
        print(f"  ✓ 2025 answer: {str(r2['answer'])[:90]}…")

        # -- window comparison ----------------------------------------------
        cmp_ = http_json("POST", f"{api_base}/api/query/compare",
                         {"question": "How did the rules change?",
                          "window_a": "2023", "window_b": "Q4 2024"})
        comp = cmp_.get("comparison") or {}
        assert comp.get("added") or comp.get("removed") or comp.get("changed"), \
            f"comparison produced no deltas: {comp}"
        print(f"  ✓ compare 2023 vs Q4 2024: +{len(comp.get('added') or [])} "
              f"-{len(comp.get('removed') or [])} ~{len(comp.get('changed') or [])}")

        # -- graph ------------------------------------------------------------
        stats = http_json("GET", f"{api_base}/api/graph/stats")
        assert stats.get("node_count", 0) >= 5 and stats.get("edge_count", 0) >= 3, \
            f"graph too small: {stats}"
        assert stats.get("temporally_validated_edges", 0) >= 1, "no interval-validated edges"
        print(f"  ✓ graph: {stats['node_count']} nodes / {stats['edge_count']} edges "
              f"({stats.get('temporally_validated_edges')} interval-validated)")

        # -- ui ---------------------------------------------------------------
        with urllib.request.urlopen(f"{api_base}/ui/", timeout=10) as res:
            assert res.status == 200 and b"Chronos" in res.read(), "dashboard not served"
        print("  ✓ dashboard served at /ui/")

        print("\nPASS — full-stack smoke succeeded.")
        return 0
    except Exception as exc:  # noqa: BLE001 - smoke must report, not crash ugly
        print(f"\nFAIL — {exc}", file=sys.stderr)
        return 1
    finally:
        for proc in procs:
            proc.terminate()
        for proc in procs:
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
        if args.keep:
            print(f"artifacts kept at {artifacts}")
        else:
            shutil.rmtree(artifacts, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
