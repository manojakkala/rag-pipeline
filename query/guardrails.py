"""
query/guardrails.py  —  Stage 6: input guardrails (prompt-injection defense).

Injection can come from the USER or from a RETRIEVED document. This flags obvious
attempts so we can refuse or strip them. It is a best-effort SECOND layer — the
main defense is architectural: the answer prompt treats context as DATA, and the
API branch is read-only.
"""
import re

PATTERNS = [
    r"ignore (all |your )?(previous |prior )?instructions",
    r"disregard (the |all )?(above|previous|prior)",
    r"forget (everything|your instructions|the above)",
    r"system prompt",
    r"reveal (all |the )?(secrets|passwords|api keys|your prompt|system prompt)",
    r"you are now",
    r"\bact as\b",
    r"print .*(secret|password|key)",
]
_RX = re.compile("|".join(PATTERNS), re.IGNORECASE)


def looks_like_injection(text: str) -> bool:
    """True if the text contains a known prompt-injection pattern (user input or a chunk)."""
    return bool(_RX.search(text or ""))
