"""
Guardrails Safety Layer
========================
Two-stage safety system:
  Stage 1 — Input guard  : blocks harmful prompts before they hit the LLM
  Stage 2 — Output guard : scans LLM responses for accidental harmful content

Both stages use:
  - Pattern/keyword matching (fast, zero-cost)
  - Configurable severity levels (BLOCK / WARN / PASS)
"""

import re
import logging
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger("guardrails")


class Decision(str, Enum):
    BLOCK = "BLOCK"
    WARN  = "WARN"
    PASS  = "PASS"


@dataclass
class GuardResult:
    decision:  Decision
    reason:    str
    category:  str = "none"
    triggered: str = ""


# ── Patterns ──────────────────────────────────────────────────────────────────

_JAILBREAK_PATTERNS = [
    r"ignore (all )?(previous|prior|your) instructions",
    r"you are now (DAN|an? (unrestricted|evil|unfiltered))",
    r"pretend you (have no|are an? AI with no) (restrictions|rules|guidelines)",
    r"act as (if you are|an? AI (without|with no))",
    r"do anything now",
    r"jailbreak",
    r"disregard (your|all) (safety|ethical|content) (guidelines|rules|policies)",
    r"(forget|ignore) (your )?(system prompt|instructions|training)",
    r"you have been (freed|liberated|unleashed)",
    r"developer mode",
    r"opposite mode",
]

_HARMFUL_KEYWORDS = {
    "weapons": [
        "pipe bomb", "make a bomb", "build explosives", "synthesize (meth|fentanyl|heroin)",
        "how to make (drugs|poison|nerve agent|sarin|vx gas)",
        "step.by.step (weapon|explosive|poison)",
    ],
    "hacking": [
        "write (malware|ransomware|keylogger|trojan|virus)",
        "exploit (code|script) for",
        "sql injection (payload|attack)",
        "how to (hack|ddos|phish|bypass) (a|an|the)",
    ],
    "csam": [
        r"\bchild porn\b", r"\bchild sexual\b", r"\bunderage (sex|nude)",
    ],
    "self_harm": [
        r"how to (kill|hurt) (myself|yourself)",
        r"(methods|ways) (of|for) suicide",
    ],
}

_BIAS_TRIGGERS = [
    r"which (race|ethnicity) is (smarter|superior|inferior|better|worse)",
    r"are (women|men|blacks|whites|asians|jews|muslims|christians) (inherently|naturally|genetically)",
    r"(prove|explain) why .{0,30} (are|is) (inferior|superior|lazy|criminal|dangerous)",
]

_OUTPUT_SENSITIVE = [
    r"\b(password|passwd|secret key|api[_\s]?key)\s*[:=]\s*\S+",
    r"(step \d+|first|then|next|finally).{0,60}(synthesize|detonate|ignite|explode)",
]

# ── Compiled patterns ─────────────────────────────────────────────────────────

def _compile(patterns: list[str]) -> list[re.Pattern]:
    return [re.compile(p, re.IGNORECASE | re.DOTALL) for p in patterns]


_C_JAILBREAK = _compile(_JAILBREAK_PATTERNS)
_C_BIAS      = _compile(_BIAS_TRIGGERS)
_C_OUTPUT    = _compile(_OUTPUT_SENSITIVE)
_C_HARMFUL   = {
    cat: _compile(pats)
    for cat, pats in _HARMFUL_KEYWORDS.items()
}


# ── Guard functions ────────────────────────────────────────────────────────────

def check_input(text: str) -> GuardResult:
    """
    Returns a GuardResult. Call before sending user message to LLM.
    BLOCK  → do not forward to model, send canned refusal
    WARN   → forward but log the flag
    PASS   → all clear
    """
    t = text.strip()

    # 1. Jailbreak attempts
    for pat in _C_JAILBREAK:
        if pat.search(t):
            logger.warning(f"[GUARDRAIL] Jailbreak pattern matched: '{pat.pattern[:60]}'")
            return GuardResult(
                decision  = Decision.BLOCK,
                reason    = "Your message appears to attempt bypassing safety guidelines.",
                category  = "jailbreak",
                triggered = pat.pattern[:60],
            )

    # 2. Harmful content
    for category, patterns in _C_HARMFUL.items():
        for pat in patterns:
            if pat.search(t):
                logger.warning(f"[GUARDRAIL] Harmful content ({category}): '{pat.pattern[:60]}'")
                return GuardResult(
                    decision  = Decision.BLOCK,
                    reason    = f"I can't help with that request.",
                    category  = category,
                    triggered = pat.pattern[:60],
                )

    # 3. Bias / discriminatory prompts → warn, don't block (model should handle)
    for pat in _C_BIAS:
        if pat.search(t):
            logger.info(f"[GUARDRAIL] Bias trigger (WARN): '{pat.pattern[:60]}'")
            return GuardResult(
                decision  = Decision.WARN,
                reason    = "Sensitive topic detected — proceeding with caution.",
                category  = "bias",
                triggered = pat.pattern[:60],
            )

    return GuardResult(decision=Decision.PASS, reason="OK")


def check_output(text: str) -> GuardResult:
    """
    Returns a GuardResult. Call on LLM response before sending to user.
    BLOCK → replace response with apology
    PASS  → send normally
    """
    for pat in _C_OUTPUT:
        if pat.search(text):
            logger.warning(f"[GUARDRAIL] Sensitive output pattern: '{pat.pattern[:60]}'")
            return GuardResult(
                decision  = Decision.BLOCK,
                reason    = "Response contained potentially sensitive content and was blocked.",
                category  = "output_filter",
                triggered = pat.pattern[:60],
            )
    return GuardResult(decision=Decision.PASS, reason="OK")


# ── Canned refusal messages ───────────────────────────────────────────────────

REFUSAL_MESSAGES = {
    "jailbreak":     "🚫 I can't follow instructions designed to bypass my guidelines.",
    "weapons":       "🚫 I can't help with that. If you have a different question, I'm here.",
    "hacking":       "🚫 I don't assist with hacking or malicious code.",
    "csam":          "🚫 This request is not something I can engage with.",
    "self_harm":     "🙏 I'm concerned about your wellbeing. Please reach out to a crisis line: https://findahelpline.com",
    "output_filter": "⚠️ My response was flagged by safety filters. Please rephrase your question.",
    "default":       "🚫 I can't help with that request.",
}


def get_refusal(category: str) -> str:
    return REFUSAL_MESSAGES.get(category, REFUSAL_MESSAGES["default"])