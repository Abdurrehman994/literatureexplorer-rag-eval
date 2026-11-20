"""Check that the Lehmus AI judge and embedding models are reachable (needs university network/VPN).

Usage:  python src/check_lehmus.py
"""
from openai import OpenAI

from config import (LEHMUS_API_KEY, LEHMUS_BASE_URL, LEHMUS_EMBED_MODEL,
                    LEHMUS_LARGE_MODEL, LEHMUS_SMALL_MODEL)

llm = OpenAI(base_url=LEHMUS_BASE_URL, api_key=LEHMUS_API_KEY, timeout=60, max_retries=0)

for name, model in [("small", LEHMUS_SMALL_MODEL), ("large", LEHMUS_LARGE_MODEL)]:
    try:
        r = llm.chat.completions.create(
            model=model, messages=[{"role": "user", "content": "Reply with the single word: OK"}])
        print(f"{name} chat model: {r.choices[0].message.content!r}")
    except Exception as exc:
        print(f"{name} chat model FAILED: {type(exc).__name__}: {exc}")

try:
    e = llm.embeddings.create(model=LEHMUS_EMBED_MODEL, input="Bananas are yellow.")
    print(f"embedding model: dim={len(e.data[0].embedding)}")
except Exception as exc:
    print(f"embedding model FAILED: {type(exc).__name__}: {exc}")
