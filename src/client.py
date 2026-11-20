"""Cached, rate-limited client for the LiteratureExplorer RAG endpoint.

Every request goes through `request()`, which
  - returns the cached record if the identical request (incl. `tag`) was made before,
  - allows at most MAX_PARALLEL requests in flight (endpoint rule: 2-3),
  - waits before retrying on 429/502/503/504 (never retries in a tight loop),
  - appends the full record (request, status, latency, response) to
    data/raw_responses/endpoint_cache.jsonl. That file is the raw-response deliverable.

Use a different `tag` to deliberately repeat a request (e.g. consistency or latency tests).
The API key is never written to disk; records only store the auth *mode*.
"""
import hashlib
import json
import threading
import time
from datetime import datetime, timezone

import requests

from config import RAG_API_KEY, RAG_BASE_URL, RAW_DIR

CACHE_FILE = RAW_DIR / "endpoint_cache.jsonl"
MAX_PARALLEL = 2
RETRY_STATUSES = {429, 502, 503, 504}
RETRY_WAITS = [5, 15, 30]  # seconds before retry 1, 2, 3
TIMEOUT = 120

_semaphore = threading.Semaphore(MAX_PARALLEL)
_file_lock = threading.Lock()
_cache: dict | None = None


def _load_cache() -> dict:
    global _cache
    if _cache is None:
        _cache = {}
        if CACHE_FILE.exists():
            with CACHE_FILE.open(encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        rec = json.loads(line)
                        _cache[rec["key"]] = rec  # later lines win (refresh)
    return _cache


def _make_key(method, path, body, auth, extra_headers, tag) -> str:
    blob = json.dumps(
        {"method": method, "path": path, "body": body, "auth": auth,
         "headers": extra_headers or {}, "tag": tag},
        sort_keys=True, ensure_ascii=False,
    )
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:20]


def _headers(auth: str, extra_headers: dict | None) -> dict:
    headers = {"Content-Type": "application/json"}
    if auth == "valid":
        headers["Authorization"] = f"Bearer {RAG_API_KEY}"
    elif auth == "invalid":
        headers["Authorization"] = "Bearer invalid-key-000000"
    elif auth != "none":
        raise ValueError(f"unknown auth mode {auth!r}")
    headers.update(extra_headers or {})
    return headers


def _send(method, path, body, headers) -> dict:
    """One HTTP attempt. Returns an attempt dict; never raises on HTTP/network errors."""
    kwargs = {"headers": headers, "timeout": TIMEOUT}
    if isinstance(body, (bytes, str)):
        kwargs["data"] = body.encode("utf-8") if isinstance(body, str) else body
    elif body is not None:
        kwargs["json"] = body
    start = time.perf_counter()
    try:
        resp = requests.request(method, RAG_BASE_URL + path, **kwargs)
        return {"status": resp.status_code, "latency_s": round(time.perf_counter() - start, 3),
                "text": resp.text, "resp_headers": dict(resp.headers), "error": None}
    except requests.RequestException as exc:
        return {"status": None, "latency_s": round(time.perf_counter() - start, 3),
                "text": None, "resp_headers": {}, "error": f"{type(exc).__name__}: {exc}"}


def request(method: str, path: str, body=None, *, auth: str = "valid",
            extra_headers: dict | None = None, tag: str = "",
            retry: bool = True, refresh: bool = False) -> dict:
    """Send (or replay from cache) one request to the endpoint and return its record."""
    cache = _load_cache()
    key = _make_key(method, path, body, auth, extra_headers, tag)
    if not refresh and key in cache:
        return cache[key]

    headers = _headers(auth, extra_headers)
    attempts = []
    with _semaphore:
        for i in range(len(RETRY_WAITS) + 1):
            attempt = _send(method, path, body, headers)
            attempts.append({k: attempt[k] for k in ("status", "latency_s", "error")})
            retryable = attempt["status"] in RETRY_STATUSES or attempt["error"] is not None
            if not (retry and retryable and i < len(RETRY_WAITS)):
                break
            time.sleep(RETRY_WAITS[i])

    try:
        resp_json = json.loads(attempt["text"]) if attempt["text"] else None
    except json.JSONDecodeError:
        resp_json = None
    record = {
        "key": key,
        "tag": tag,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "method": method,
        "path": path,
        "request_body": body,
        "auth": auth,
        "extra_headers": extra_headers or {},
        "status": attempt["status"],
        "latency_s": attempt["latency_s"],
        "error": attempt["error"],
        "attempts": attempts,
        "response_json": resp_json,
        "response_text": None if resp_json is not None else attempt["text"],
        "response_headers": attempt["resp_headers"],
    }
    with _file_lock:
        with CACHE_FILE.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        cache[key] = record
    return record


def context(query: str, tag: str = "", **kw) -> dict:
    """Retrieval only: POST /v1/context."""
    return request("POST", "/v1/context", {"query": query}, tag=tag, **kw)


def chat(question: str, tag: str = "", **kw) -> dict:
    """Retrieval + generation: POST /v1/chat/completions (single user turn)."""
    body = {"messages": [{"role": "user", "content": question}]}
    return request("POST", "/v1/chat/completions", body, tag=tag, **kw)


def answer_text(record: dict) -> str | None:
    """Extract the assistant message from a chat-completions record."""
    try:
        return record["response_json"]["choices"][0]["message"]["content"]
    except (TypeError, KeyError, IndexError):
        return None


def usage() -> dict:
    """HTTP calls actually sent so far (counting retries), for the ~1000-request budget."""
    calls = 0
    records = 0
    if CACHE_FILE.exists():
        with CACHE_FILE.open(encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    records += 1
                    calls += len(json.loads(line)["attempts"])
    return {"records": records, "http_calls": calls}


if __name__ == "__main__":
    print(usage())
