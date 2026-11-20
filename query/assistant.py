"""
query/assistant.py  —  Stage 6: the unified entry point.

ask(question) routes every question to the right path:
  RAG     -> cached_answer  (docs: cache -> rewrite -> hybrid -> rerank -> grounded answer)
  API     -> answer_api     (live GitHub data)
  DECLINE -> polite refusal (out of scope)

This is the "front desk" from the routing lesson, now wired to the departments.
"""
import sys

sys.stdout.reconfigure(encoding="utf-8")

from router import route
from github_api import answer_api
from generate import cached_answer, chat_reply
from guardrails import looks_like_injection


def ask(question: str) -> str:
    if looks_like_injection(question):      # input guardrail
        return "[blocked] That looks like a prompt-injection attempt, so I won't act on it."
    r = route(question)                     # the front desk decides the department
    if r == "API":
        return f"[route: API]\n{answer_api(question)}"
    if r == "CHAT":
        return f"[route: CHAT]\n{chat_reply(question)}"
    if r == "DECLINE":
        return ("[route: DECLINE]\nThat's outside what I can help with — "
                "I answer GitHub documentation questions and live GitHub data.")
    return f"[route: RAG]\n{cached_answer(question)}"


if __name__ == "__main__":
    q = sys.argv[1] if len(sys.argv) > 1 else "How many open issues does github/docs have right now?"
    print(f"Q: {q}\n")
    print(ask(q))
