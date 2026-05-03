"""
End-to-end MCP protocol test: spawns the server as a subprocess over stdio,
talks to it as a real MCP client would, and exercises the Phase 1 tools.

This is what Claude Desktop / Continue / Goose / etc. all do under the hood —
if this passes, the server works for any MCP-compliant client.
"""
import asyncio
import json
from datetime import timedelta
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


ROOT = Path(__file__).resolve().parent
PYTHON = ROOT / ".venv" / "Scripts" / "python.exe"
SAMPLE = (ROOT.parent / "data" / "sample.csv").resolve()


async def main() -> None:
    params = StdioServerParameters(
        command=str(PYTHON),
        args=["-m", "curator_mcp.server"],
        cwd=str(ROOT),
    )
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            print("[1/3] list_tools")
            tools = await session.list_tools()
            names = [t.name for t in tools.tools]
            print(f"      tools={names}")
            assert "load" in names and "status" in names

            print("[2/3] call status (before load)")
            r = await session.call_tool("status", {})
            payload = json.loads(r.content[0].text)
            print(f"      {payload}")
            assert payload["loaded"] is False

            print("[3/3] call load + status (after)")
            # First load in a fresh subprocess pays numba JIT cost on UMAP fit; bump timeout.
            r = await session.call_tool(
                "load",
                {"path": str(SAMPLE), "text_column": "text", "reduce_to_2d": False},
                read_timeout_seconds=timedelta(seconds=120),
            )
            summary = json.loads(r.content[0].text)
            print(f"      load summary: rows={summary['rows']}, dim={summary['embedding_dim']}, "
                  f"hits={summary['embeddings_cached_hits']}")
            assert summary["rows"] == 18
            assert summary["embedding_dim"] == 1024
            # Second-time-through cache should be hot from the previous smoke test.
            assert summary["embeddings_cached_hits"] == 18

            r = await session.call_tool("status", {})
            st = json.loads(r.content[0].text)
            assert st["loaded"] is True
            assert st["kept_count"] == 18

    print("OK - MCP server speaks the protocol; load/status work over stdio.")


if __name__ == "__main__":
    asyncio.run(main())
