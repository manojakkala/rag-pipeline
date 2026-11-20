"""
query/router.py  —  Stage 6: query ROUTING.

Classify a question into the right path BEFORE any retrieval happens:
  RAG     - answerable from the GitHub documentation (how-to, concepts, policies)
  API     - needs LIVE data from GitHub right now (counts, status, a repo's current state)
  DECLINE - unrelated to GitHub / out of scope

One cheap LLM call (routing is an easy task → small model).
"""
import os

from openai import OpenAI
from dotenv import load_dotenv

load_dotenv(".env.local")

client = OpenAI(api_key=os.environ["GROQ_API_KEY"], base_url="https://api.groq.com/openai/v1")
MODEL = os.environ.get("GROQ_SMALL_MODEL", "openai/gpt-oss-20b")  # cheap/fast model

ROUTER_SYSTEM = (
    "You are a router for a GitHub documentation assistant. "
    "Classify the user's message into EXACTLY one label:\n"
    "RAG - answerable by reading GitHub documentation (how-to, concepts, policies, API usage).\n"
    "API - needs LIVE data from GitHub right now (e.g. a repo's open-issue count, stars, latest release).\n"
    "CHAT - a greeting, thanks, small talk, or a question about what you can do / how you can help.\n"
    "DECLINE - unrelated to GitHub and NOT a greeting (e.g. weather, jokes, cooking).\n"
    "Reply with ONLY the label: RAG, API, CHAT, or DECLINE."
)

VALID = ("RAG", "API", "CHAT", "DECLINE")


def route(question: str) -> str:
    resp = client.chat.completions.create(
        model=MODEL,
        temperature=0,
        messages=[
            {"role": "system", "content": ROUTER_SYSTEM},
            {"role": "user", "content": question},
        ],
    )
    label = (resp.choices[0].message.content or "").strip().upper()
    for v in VALID:
        if v in label:
            return v
    return "RAG"  # safe default: if unsure, try the docs rather than refuse


if __name__ == "__main__":
    tests = [
        "What is GitHub's REST API versioning policy?",
        "How do I store secrets in GitHub Actions?",
        "How many open issues does the github/docs repository have right now?",
        "How many stars does the facebook/react repo have?",
        "What's the weather today?",
        "Write me a haiku about continuous integration.",
    ]
    for q in tests:
        print(f"{route(q):8} <- {q}")
