"""
Observability Module
=====================
Structured JSON logging for every LLM interaction.
Captures: timestamp, model, latency, token estimates,
          RAG context used, guardrail decisions, user_id (hashed).

Log file: logs/interactions.jsonl  (one JSON object per line)

Usage:
    from shared.observability import log_interaction, get_stats

    log_interaction(
        model="Qwen/Qwen2.5-7B-Instruct",
        user_id=12345678,
        prompt="What is Ollive?",
        response="Ollive is an AI liability insurance company...",
        latency_sec=3.2,
        rag_chunks_used=3,
        guardrail_decision="PASS",
        guardrail_category="none",
    )
"""

import json
import hashlib
import logging
import os
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger("observability")

LOG_DIR  = Path(__file__).resolve().parent.parent / "logs"
LOG_FILE = LOG_DIR / "interactions.jsonl"

LOG_DIR.mkdir(exist_ok=True)


# ── Token estimator (rough: 1 token ≈ 4 chars) ───────────────────────────────

def _estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def _hash_user(user_id: int) -> str:
    """One-way hash of user_id for privacy."""
    return hashlib.sha256(str(user_id).encode()).hexdigest()[:12]


# ── Core logging function ─────────────────────────────────────────────────────

def log_interaction(
    *,
    model:               str,
    user_id:             int,
    prompt:              str,
    response:            str,
    latency_sec:         float,
    rag_chunks_used:     int   = 0,
    rag_sources:         list  = None,
    guardrail_decision:  str   = "PASS",
    guardrail_category:  str   = "none",
    error:               str   = None,
    bot_type:            str   = "unknown",
) -> dict:
    """Write one structured log entry to logs/interactions.jsonl."""

    prompt_tokens   = _estimate_tokens(prompt)
    response_tokens = _estimate_tokens(response) if response else 0

    entry = {
        "ts":                  datetime.now(timezone.utc).isoformat(),
        "bot_type":            bot_type,
        "model":               model,
        "user_id_hash":        _hash_user(user_id),
        "latency_sec":         round(latency_sec, 3),
        "prompt_tokens_est":   prompt_tokens,
        "response_tokens_est": response_tokens,
        "total_tokens_est":    prompt_tokens + response_tokens,
        "rag_chunks_used":     rag_chunks_used,
        "rag_sources":         rag_sources or [],
        "guardrail_decision":  guardrail_decision,
        "guardrail_category":  guardrail_category,
        "error":               error,
        "prompt_preview":      prompt[:120],
        "response_preview":    response[:120] if response else "",
    }

    try:
        with open(LOG_FILE, "a") as f:
            f.write(json.dumps(entry) + "\n")
    except Exception as e:
        logger.error(f"Failed to write observability log: {e}")

    logger.debug(f"[OBS] {bot_type} | {model} | {latency_sec:.2f}s | "
                 f"RAG:{rag_chunks_used} | Guard:{guardrail_decision}")

    return entry


# ── Stats aggregator ──────────────────────────────────────────────────────────

def get_stats(bot_type: str = None) -> dict:
    """
    Read all log entries and return aggregate stats.
    Optionally filter by bot_type ('oss' or 'frontier').
    """
    if not LOG_FILE.exists():
        return {"error": "No log file found. Run the bots first."}

    entries = []
    with open(LOG_FILE) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

    if bot_type:
        entries = [e for e in entries if e.get("bot_type") == bot_type]

    if not entries:
        return {"total_interactions": 0}

    latencies  = [e["latency_sec"] for e in entries if e.get("latency_sec")]
    tokens     = [e["total_tokens_est"] for e in entries if e.get("total_tokens_est")]
    rag_used   = [e for e in entries if e.get("rag_chunks_used", 0) > 0]
    blocked    = [e for e in entries if e.get("guardrail_decision") == "BLOCK"]
    errors     = [e for e in entries if e.get("error")]

    return {
        "total_interactions":   len(entries),
        "avg_latency_sec":      round(sum(latencies) / len(latencies), 3) if latencies else 0,
        "p95_latency_sec":      round(sorted(latencies)[int(len(latencies) * 0.95)], 3) if latencies else 0,
        "avg_tokens_per_call":  round(sum(tokens) / len(tokens)) if tokens else 0,
        "total_tokens_est":     sum(tokens),
        "rag_usage_rate":       round(len(rag_used) / len(entries), 3),
        "guardrail_block_rate": round(len(blocked) / len(entries), 3),
        "error_rate":           round(len(errors) / len(entries), 3),
        "blocked_count":        len(blocked),
        "error_count":          len(errors),
    }