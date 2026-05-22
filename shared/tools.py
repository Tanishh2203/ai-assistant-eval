"""
Tool Use Module
================
Simple tools available to both bots:

  - calculator   : evaluate safe math expressions
  - get_datetime : current date and time
  - kb_search    : search the RAG knowledge base explicitly

Tool routing is keyword-based (no function-calling API required),
so it works with any model including OSS ones that don't support
native function calling.
"""

import re
import logging
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger("tools")


# ── Tool definitions ──────────────────────────────────────────────────────────

TOOL_TRIGGERS = {
    "calculator": [
        r"(calculate|compute|what is|solve|evaluate)\s+[\d\s\+\-\*\/\(\)\.\^%]+",
        r"\d+\s*[\+\-\*\/\^]\s*\d+",
        r"(what'?s|whats)\s+\d+\s*(plus|minus|times|divided by|multiplied by)\s*\d+",
    ],
    "get_datetime": [
        r"(what|tell me).{0,20}(today'?s? date|current (date|time)|date today|time (now|today))",
        r"(what|which) (day|month|year) is (it|today)",
        r"(current|today'?s?) (date|time|day)",
    ],
    "kb_search": [
        r"(search|look up|find|what does (the )?(doc|knowledge base|kb) say)",
        r"according to (the )?(document|knowledge base|kb|your data)",
    ],
}

_COMPILED_TRIGGERS = {
    tool: [re.compile(p, re.IGNORECASE) for p in patterns]
    for tool, patterns in TOOL_TRIGGERS.items()
}


def detect_tool(message: str) -> Optional[str]:
    """Return the name of the tool to invoke, or None."""
    for tool, patterns in _COMPILED_TRIGGERS.items():
        for pat in patterns:
            if pat.search(message):
                return tool
    return None


# ── Tool implementations ──────────────────────────────────────────────────────

def run_calculator(expression: str) -> str:
    """
    Safely evaluate a math expression.
    Only allows digits, operators, parentheses, and whitespace.
    """
    # Extract the numeric expression from the message
    match = re.search(
        r"([\d\s\+\-\*\/\(\)\.\^%]+(?:[\d\)])+)",
        expression
    )
    if not match:
        return None

    raw = match.group(1).strip()
    # Replace ^ with ** for exponentiation
    safe = raw.replace("^", "**")

    # Whitelist: only allow digits, operators, spaces, parens, dots
    if not re.fullmatch(r"[\d\s\+\-\*\/\(\)\.\*%]+", safe):
        return None

    try:
        result = eval(safe, {"__builtins__": {}})  # noqa: S307
        return f"🧮 {raw} = **{result}**"
    except Exception as e:
        logger.debug(f"Calculator eval error: {e}")
        return None


def run_get_datetime() -> str:
    """Return current UTC date and time."""
    now = datetime.now(timezone.utc)
    return (
        f"🕐 Current date/time (UTC): **{now.strftime('%A, %B %d, %Y at %H:%M')}**"
    )


def run_kb_search(query: str, rag_engine) -> str:
    """Explicitly search the knowledge base and return top results."""
    if rag_engine is None:
        return None
    chunks, metas = rag_engine.retrieve(query, top_k=3)
    if not chunks:
        return "📚 No relevant documents found in the knowledge base."
    sources = sorted({m.get("source", "?") for m in metas})
    result  = "📚 **Knowledge base results:**\n\n"
    for i, chunk in enumerate(chunks, 1):
        result += f"**[{i}]** {chunk[:300]}…\n\n"
    result += f"_Sources: {', '.join(sources)}_"
    return result


# ── Main dispatcher ───────────────────────────────────────────────────────────

def try_tool(message: str, rag_engine=None) -> Optional[str]:
    """
    Try to handle a message with a tool.
    Returns a tool response string, or None if no tool matched.
    """
    tool = detect_tool(message)
    if tool is None:
        return None

    logger.info(f"[TOOL] Routing to: {tool}")

    if tool == "calculator":
        return run_calculator(message)

    if tool == "get_datetime":
        return run_get_datetime()

    if tool == "kb_search":
        return run_kb_search(message, rag_engine)

    return None 