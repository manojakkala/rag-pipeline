"""
query/github_api.py  —  Stage 6: the LIVE API branch.

For questions the router sends to "API" (live data, not docs), we:
  1. extract which repo the question is about (owner/repo)
  2. call GitHub's public REST API (read-only GET)
  3. hand the live JSON to the model to phrase a grounded answer

Safety: read-only public GET only. The model only READS live data, never acts.
"""
import json
import os
import re
import urllib.error
import urllib.request

from openai import OpenAI
from dotenv import load_dotenv

load_dotenv(".env.local")

client = OpenAI(api_key=os.environ["GROQ_API_KEY"], base_url="https://api.groq.com/openai/v1")
MODEL = os.environ.get("GROQ_MODEL", "openai/gpt-oss-120b")
SMALL = os.environ.get("GROQ_SMALL_MODEL", "openai/gpt-oss-20b")

FIELDS = ("full_name", "description", "open_issues_count", "stargazers_count",
          "forks_count", "language", "updated_at")


def extract_repo(question: str):
    """Pull the repository as 'owner/repo' from the question (cheap model), or None."""
    resp = client.chat.completions.create(
        model=SMALL, temperature=0,
        messages=[
            {"role": "system", "content": "Extract the GitHub repository as 'owner/repo' "
             "from the question. Reply ONLY with owner/repo, or the word none."},
            {"role": "user", "content": question},
        ],
    )
    v = (resp.choices[0].message.content or "").strip()
    return v if ("/" in v and v.lower() != "none") else None


def fetch_repo(owner_repo: str):
    """Read-only GET to the public GitHub API. Returns (data, error)."""
    if not re.fullmatch(r"[\w.-]+/[\w.-]+", owner_repo):      # only owner/repo — no path tricks
        return None, "invalid repository name"
    url = f"https://api.github.com/repos/{owner_repo}"
    req = urllib.request.Request(
        url, headers={"User-Agent": "rag-assistant", "Accept": "application/vnd.github+json"}
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            d = json.load(r)
    except urllib.error.HTTPError as e:
        return None, f"HTTP {e.code}"
    except Exception as e:  # network, timeout, etc.
        return None, str(e)
    return {k: d.get(k) for k in FIELDS}, None


def answer_api(question: str) -> str:
    repo = extract_repo(question)
    if not repo:
        return "This needs live data, but I couldn't find a repository (owner/repo) in your question."

    data, err = fetch_repo(repo)
    if err:
        return f"Couldn't reach the GitHub API for '{repo}' ({err})."

    out = client.chat.completions.create(
        model=MODEL, temperature=0,
        messages=[
            {"role": "system", "content": "Answer the question using ONLY this live GitHub API "
             "data (JSON). Be concise. If the data doesn't contain the answer, say so."},
            {"role": "user", "content": f"Live data for {repo}:\n{json.dumps(data)}\n\nQuestion: {question}"},
        ],
    ).choices[0].message.content

    return f"{out}\n\n[source: live GitHub API — /repos/{repo}]"
