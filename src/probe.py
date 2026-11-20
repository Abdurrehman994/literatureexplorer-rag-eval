"""First look at the endpoint: what do both routes return? (~4 endpoint requests, cached.)

Usage:  python src/probe.py
"""
import json

import client

QUESTIONS = [
    "What is WOLF?",                                   # example from the assignment
    "What is the capital of Australia?",               # clearly out of scope
]





def show(title: str, record: dict, max_chars: int = 2500) -> None:
    print(f"\n=== {title} | HTTP {record['status']} | {record['latency_s']} s ===")
    body = record["response_json"] if record["response_json"] is not None else record["response_text"]
    text = json.dumps(body, indent=2, ensure_ascii=False) if not isinstance(body, str) else body
    print(text[:max_chars] + ("\n... [truncated]" if text and len(text) > max_chars else ""))


for q in QUESTIONS:
    show(f"context: {q}", client.context(q, tag="probe"))
    show(f"chat: {q}", client.chat(q, tag="probe"))

print("\nBudget used:", client.usage())
