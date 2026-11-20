"""
query/rewrite.py  —  Query rewriting (part of "query understanding").

Rephrase/expand the user's question into a better SEARCH query using the LLM,
so vocabulary-gap questions ("run on my own servers") match the docs' own words
("self-hosted runners"). One small LLM call, then we search with the result.
"""
import os
import re

from openai import OpenAI
from dotenv import load_dotenv

load_dotenv(".env.local")

client = OpenAI(api_key=os.environ["GROQ_API_KEY"], base_url="https://api.groq.com/openai/v1")
MODEL = os.environ.get("GROQ_SMALL_MODEL", "openai/gpt-oss-20b")  # rewriting is easy → cheap/fast model

REWRITE_SYSTEM = (
    "Rewrite the user's question into a concise search query for technical documentation. "
    "Expand it with the likely official terminology and synonyms "
    "(for example, 'my own servers' -> 'self-hosted runners'; "
    "'without long-lived keys' -> 'OpenID Connect OIDC'). "
    "Return ONLY the rewritten query text, nothing else."
)


def rewrite_query(question: str) -> str:
    resp = client.chat.completions.create(
        model=MODEL,
        temperature=0,
        messages=[
            {"role": "system", "content": REWRITE_SYSTEM},
            {"role": "user", "content": question},
        ],
    )
    text = resp.choices[0].message.content or ""
    # LLM output can carry stray control/NUL bytes — strip them before the text
    # is used as a SQL query (Postgres text cannot contain NUL) or printed.
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", " ", text)
    return " ".join(text.split()).strip()
