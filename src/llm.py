"""Cached calls to the Lehmus AI chat models (test-set generation and judging).

Every call is cached in data/llm_cache.jsonl, keyed by (model, messages, temperature), so reruns
cost nothing and results are reproducible. Not an endpoint request (no budget impact).
"""
import hashlib
import json
import re
import threading

from openai import OpenAI

from config import (DATA_DIR, LEHMUS_API_KEY, LEHMUS_BASE_URL, LEHMUS_LARGE_MODEL,
                    LEHMUS_SMALL_MODEL, require)

SMALL = LEHMUS_SMALL_MODEL   # Gemma 4 26B MoE: bulk work
LARGE = LEHMUS_LARGE_MODEL   # GPT-OSS 120B: quality-critical work
CACHE_FILE = DATA_DIR / "llm_cache.jsonl"

_client: OpenAI | None = None  # created on first real call, so cached replays need no .env
_lock = threading.Lock()
_cache: dict | None = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(base_url=require("LEHMUS_BASE_URL", LEHMUS_BASE_URL),
                         api_key=require("LEHMUS_API_KEY", LEHMUS_API_KEY), timeout=300, max_retries=3)
    return _client


def _load() -> dict:
    """Load the cache once. Built fully before publishing it, so no thread sees a half-loaded cache."""
    global _cache
    with _lock:
        if _cache is None:
            loaded = {}
            if CACHE_FILE.exists():
                for line in CACHE_FILE.open(encoding="utf-8"):
                    if line.strip():
                        rec = json.loads(line)
                        loaded[rec["key"]] = rec["output"]
            _cache = loaded
    return _cache


def chat(prompt: str, model: str = SMALL, temperature: float = 0.0, system: str | None = None) -> str:
    messages = ([{"role": "system", "content": system}] if system else []) + \
               [{"role": "user", "content": prompt}]
    key = hashlib.sha256(json.dumps([model, messages, temperature]).encode()).hexdigest()[:24]
    cache = _load()
    if key in cache:
        return cache[key]
    resp = _get_client().chat.completions.create(model=model, messages=messages, temperature=temperature)
    output = resp.choices[0].message.content or ""
    with _lock:
        with CACHE_FILE.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"key": key, "model": model, "messages": messages,
                                "temperature": temperature, "output": output},
                               ensure_ascii=False) + "\n")
        cache[key] = output
    return output


def parse_json(text: str) -> dict | None:
    """Parse a JSON object from a model reply (tolerates ```json fences and surrounding text)."""
    match = re.search(r"\{.*\}", text, flags=re.S)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


def chat_json(prompt: str, model: str = SMALL, temperature: float = 0.0) -> dict | None:
    out = parse_json(chat(prompt, model, temperature))
    if out is None:  # one retry with an explicit reminder (different prompt -> different cache key)
        out = parse_json(chat(prompt + "\n\nReturn ONLY a valid JSON object.", model, temperature))
    return out
