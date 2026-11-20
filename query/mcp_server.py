"""
query/mcp_server.py  —  Stage 9: expose the whole assistant as an MCP server.

Any MCP client (e.g. Claude Desktop) can then call `ask_rag_assistant` as a tool,
which runs the full pipeline (router -> RAG or live GitHub API -> grounded answer).

Run as:  .venv\\Scripts\\python.exe query\\mcp_server.py   (speaks MCP over stdio)
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
os.chdir(ROOT)              # so .env.local and data/ resolve no matter who launches us
sys.path.insert(0, HERE)   # so `import assistant` works

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("rag-assistant")


def ask_rag_assistant(question: str) -> str:
    """Answer a question from the GitHub documentation, or fetch live GitHub data
    (e.g. a repo's open-issue count). Returns a grounded, cited answer."""
    from assistant import ask
    return ask(question)


mcp.tool()(ask_rag_assistant)   # register the function as a callable MCP tool


if __name__ == "__main__":
    mcp.run()                   # talk MCP over stdio to a connected client
