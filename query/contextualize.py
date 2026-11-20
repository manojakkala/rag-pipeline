"""
query/contextualize.py  —  make a follow-up question STANDALONE using the chat history.

So "what about python?" asked right after an auth question becomes
"How do I authenticate to the GitHub REST API with Python?" — which then routes and
retrieves correctly. One cheap LLM call, only when there is history. Fails safe: on any
problem it returns the original question unchanged.
"""
import os
import re

from openai import OpenAI
from dotenv import load_dotenv

load_dotenv(".env.local")

client = OpenAI(api_key=os.environ["GROQ_API_KEY"], base_url="https://api.groq.com/openai/v1")
MODEL = os.environ.get("GROQ_SMALL_MODEL", "openai/gpt-oss-20b")  # easy task -> cheap/fast model

CONTEXT_SYSTEM = (
    "You rewrite a user's follow-up question into a STANDALONE question using the conversation. "
    "Resolve references like 'it', 'that', 'this', 'they', 'the same' to what they point to. "
    "If the question is already standalone, return it unchanged. "
    "Return ONLY the rewritten question, nothing else."
)


def contextualize(history, question: str) -> str:
    """history: list of {'role': 'user'|'assistant', 'content': str} (recent turns).
    Returns a standalone version of `question`; on any issue returns it unchanged."""
    if not history:
        return question
    convo = "\n".join(f"{m.get('role')}: {m.get('content')}" for m in history[-6:])
    try:
        resp = client.chat.completions.create(
            model=MODEL,
            temperature=0,
            messages=[
                {"role": "system", "content": CONTEXT_SYSTEM},
                {"role": "user",
                 "content": f"Conversation so far:\n{convo}\n\nFollow-up: {question}\n\nStandalone question:"},
            ],
        )
        text = resp.choices[0].message.content or ""
        text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", " ", text)   # strip control/NUL bytes
        return " ".join(text.split()).strip() or question
    except Exception:
        return question    # fail safe: never break the request over contextualization
