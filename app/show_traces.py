"""Run one question down each path (docs / live API / chat / decline) and print its trace,
then confirm the MCP tool is registered."""
import os
import sys

sys.stdout.reconfigure(encoding="utf-8")
from server import traced_ask   # app/server.py


def show(q):
    r = traced_ask(q)
    print("=" * 72)
    print("Q:", q)
    print(f"route: {r['route']}   grounded: {r['grounded']}")
    print("answer:", r["answer"].split("Sources")[0].strip()[:200])
    print("trace:")
    for s in r["spans"]:
        extra = ""
        if "decision" in s:  extra = "  -> " + s["decision"]
        if "rewritten" in s: extra = '  rewritten="' + s["rewritten"][:60] + '"'
        if "chunks" in s:    extra = "  top docs=" + str([(c["doc"], c["score"]) for c in s["chunks"][:3]])
        if "grounded" in s:  extra = "  grounded=" + str(s["grounded"])
        print(f"   {s['stage']:16} {s['ms']:>6} ms{extra}")
    if r["sources"]:
        print("sources:", r["sources"][:3])


for q in [
    "What is the difference between the REST API and the GraphQL API?",  # DOCS (RAG)
    "How many open issues does the github/docs repository have right now?",  # LIVE API
    "what can you do",                                                   # CHAT
    "what is the weather today",                                         # DECLINE
]:
    show(q)

print("=" * 72)
print("MCP:")
import asyncio
import mcp_server
try:
    tools = asyncio.run(mcp_server.mcp.list_tools())
    print("  MCP tool registered:", [t.name for t in tools])
    print("  (Claude Desktop calls this tool, which runs the exact same pipeline above)")
except Exception as e:
    print("  note:", e)
