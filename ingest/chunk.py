"""
ingest/chunk.py  —  Stage 3: STRUCTURE-AWARE chunking for Markdown.

Replaces the baseline paragraph splitter. For each document it:
  1. strips the YAML frontmatter block (keeps its `title:` to label chunks)
  2. removes Liquid template noise ( {% ... %} and {{ ... }} )
  3. splits on Markdown headings (#, ##, ...) into sections
  4. keeps fenced ``` code blocks whole (never splits inside one)
  5. prepends the heading path ("Secrets > About secrets:") to every chunk
Sections bigger than MAX_CHARS are packed by paragraph, code blocks kept intact.

Run:  .venv\\Scripts\\python.exe ingest\\chunk.py   (then re-run embed.py)
Idempotent: delete-before-insert per document.
"""
import os
import re
from pathlib import Path

import psycopg
from dotenv import load_dotenv

load_dotenv(".env.local")
conn_str = os.environ["DATABASE_URL"]

MAX_CHARS = 1500

FRONTMATTER = re.compile(r"^---\n.*?\n---\n", re.DOTALL)        # the ---...--- block at the top
LIQUID = re.compile(r"\{%.*?%\}|\{\{.*?\}\}", re.DOTALL)         # {% ... %} and {{ ... }}
TITLE = re.compile(r"^title:\s*['\"]?(.+?)['\"]?\s*$", re.MULTILINE)


def get_title(raw: str) -> str | None:
    m = TITLE.search(raw)
    return m.group(1).strip() if m else None


def clean(raw: str) -> str:
    raw = FRONTMATTER.sub("", raw, count=1)   # drop the frontmatter block
    raw = LIQUID.sub("", raw)                  # drop the template tags
    return raw


def segments(body: str) -> list[str]:
    """Break a section body into paragraphs, but keep fenced code blocks whole."""
    parts = re.split(r"(```.*?```)", body, flags=re.DOTALL)
    segs = []
    for p in parts:
        if p.startswith("```"):
            segs.append(p.strip())                 # a whole code block = one atomic segment
        else:
            segs += [para.strip() for para in p.split("\n\n") if para.strip()]
    return segs


def pack(segs: list[str], max_chars: int) -> list[str]:
    """Greedily combine segments up to max_chars; never split a code block."""
    chunks, cur = [], ""
    for seg in segs:
        if len(seg) > max_chars and not seg.startswith("```"):
            if cur:
                chunks.append(cur); cur = ""
            for i in range(0, len(seg), max_chars):   # hard-split an oversized plain paragraph
                chunks.append(seg[i:i + max_chars])
        elif len(cur) + len(seg) + 2 <= max_chars:
            cur = (cur + "\n\n" + seg).strip()
        else:
            if cur:
                chunks.append(cur)
            cur = seg                                 # fits alone, or oversized code kept whole
    if cur:
        chunks.append(cur)
    return chunks


def chunk_document(doc_id: str, raw: str) -> list[str]:
    title = get_title(raw) or doc_id
    text = clean(raw)

    # Walk lines, tracking code-fence state so a '#' inside code isn't seen as a heading.
    sections, cur_heading, cur_lines, in_code = [], "", [], False
    for line in text.split("\n"):
        if line.strip().startswith("```"):
            in_code = not in_code
            cur_lines.append(line)
        elif not in_code and re.match(r"^#{1,6}\s+", line):
            if cur_lines:
                sections.append((cur_heading, "\n".join(cur_lines)))
            cur_heading, cur_lines = line.lstrip("#").strip(), []
        else:
            cur_lines.append(line)
    if cur_lines:
        sections.append((cur_heading, "\n".join(cur_lines)))

    out = []
    for heading, body in sections:
        body = body.strip()
        if not body:
            continue
        label = f"{title} > {heading}" if heading else title
        for piece in pack(segments(body), MAX_CHARS):
            out.append(f"{label}:\n{piece}")        # prepend the heading path for context
    return out


with psycopg.connect(conn_str) as conn:
    with conn.cursor() as cur:
        cur.execute("SELECT doc_id, raw_text FROM documents ORDER BY doc_id")
        docs = cur.fetchall()
        total = 0
        for doc_id, raw in docs:
            cur.execute("DELETE FROM chunks WHERE doc_id = %s", (doc_id,))  # clears old chunks + their vectors
            for idx, chunk_text in enumerate(chunk_document(doc_id, raw)):
                cur.execute(
                    "INSERT INTO chunks (chunk_id, doc_id, chunk_index, text, n_chars) "
                    "VALUES (%s, %s, %s, %s, %s)",
                    (f"{doc_id}::{idx}", doc_id, idx, chunk_text, len(chunk_text)),
                )
                total += 1

print(f"Wrote {total} structure-aware chunks from {len(docs)} documents.")
print("Next: re-run embed.py to embed the new chunks.")
