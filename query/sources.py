"""
query/sources.py  —  turn a corpus doc_id into a public GitHub Docs URL for citations.

The corpus filenames mirror github/docs content/rest paths with '__' standing in for '/',
so 'actions__secrets.md' maps to https://docs.github.com/en/rest/actions/secrets .
Verified: every real doc resolves; only the corpus README has no public page.
"""

BASE = "https://docs.github.com/en/rest/"


def source_url(doc_id: str):
    """doc_id like 'actions__secrets.md' -> the GitHub Docs page URL.
    Returns None when there is no public page (e.g. the corpus README)."""
    base = doc_id[:-3] if doc_id.endswith(".md") else doc_id
    if base.lower() == "readme":
        return None
    path = base.replace("__", "/")          # flat filename -> nested path
    if path.endswith("/index"):             # an index file is the section's root page
        path = path[: -len("/index")]
    return BASE + path
