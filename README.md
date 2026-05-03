<div align="center">

# Whittle

### A local-first, LLM-driven workspace for *paring down* text datasets

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![MCP](https://img.shields.io/badge/protocol-MCP-7c3aed.svg)](https://modelcontextprotocol.io/)
[![Local-first](https://img.shields.io/badge/inference-local-22c55e.svg)](https://ollama.com)
[![Privacy](https://img.shields.io/badge/privacy-100%25%20offline-1f2937.svg)](#)
[![Status](https://img.shields.io/badge/phase%204-complete-brightgreen.svg)](#roadmap)

*Embed locally · Project to 2D · Curate by conversation · Save the kept subset*

</div>

---

> **Why this exists.** The bottleneck in modern ML is no longer model architecture — it's data quality. But every "data-centric AI" tool ships your text to a SaaS dashboard. This project keeps the entire loop on your laptop: embeddings via Ollama, dimensionality reduction via UMAP, an in-browser scatter view via Spotlight, and an MCP server so any LLM client (Claude Desktop, Claude Code, Continue, Goose) can drive the whole curation workflow in plain English.

## Contents

- [What it is](#what-it-is)
- [The two ways to use it](#the-two-ways-to-use-it)
- [Architecture](#architecture)
- [Quickstart](#quickstart)
- [MCP tools](#mcp-tools)
- [How it was built — vibe coding the whole stack](#how-it-was-built--vibe-coding-the-whole-stack)
- [Roadmap](#roadmap)
- [Project layout](#project-layout)

---

## What it is

A two-mode workspace for **manually curating text datasets** before fine-tuning, RAG indexing, or eval-set construction:

| Mode | Driver | Best for |
|------|--------|----------|
| **CLI** ([`curate.py`](curate.py)) | You | One-off datasets, scripted pipelines |
| **MCP server** ([`mcp_server/`](mcp_server/)) | An LLM client | Conversational, iterative curation |

In both modes the unit of work is the same: load → embed → project → inspect → mark → save the kept subset.

## The two ways to use it

### 1. CLI mode — fast, scripted

```bash
python curate.py --input data/sample.csv --text-column text
```

Embeds every row through your local Ollama daemon, runs UMAP, opens Spotlight in your browser. You edit `keep` / `flag` / `notes` columns in the `.curated.csv` it writes alongside the source, close the tab, and the script writes a `.kept.csv` of the surviving rows.

### 2. MCP mode — conversational, LLM-driven

Wire the server into any MCP-compatible client and curate by talking to it:

> *"Load `data/sample.csv`. Find the 5 most isolated rows. Show me near-duplicates above 0.95 similarity. Drop rows 11, 15, 16, 17 — they're lorem-ipsum noise. Flag rows 6, 7, 12 as spam suspects. Save the kept subset."*

The LLM picks the tools, you stay in natural language, and the dataset gets molded through dialogue.

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                      Your machine — 100% local                       │
│                                                                      │
│   ┌─────────────────┐     ┌──────────────────┐    ┌──────────────┐  │
│   │  LLM client     │ MCP │  curator-mcp     │    │   Ollama     │  │
│   │  Claude Code /  │◄───►│  (FastMCP +      │───►│   bge-m3     │  │
│   │  Desktop / etc. │stdio│   pandas + UMAP) │    │   1024-dim   │  │
│   └─────────────────┘     └────────┬─────────┘    └──────────────┘  │
│                                    │                                 │
│                                    │ launch_viewer                   │
│                                    ▼                                 │
│                           ┌────────────────────┐                     │
│                           │  Spotlight (browser)│                    │
│                           │  UMAP scatter +    │                     │
│                           │  inspector + table │                     │
│                           └────────────────────┘                     │
└─────────────────────────────────────────────────────────────────────┘
```

**Key design choices**

- **No SaaS, no API keys.** Ollama for embeddings, optional OpenAI-compatible backend (LM Studio, vLLM, llama.cpp server) — your text never leaves the machine.
- **Caching is non-negotiable.** Every embedding is keyed by `SHA256(model || text)` and stored in a SQLite file beside the dataset. Re-runs and "load again" calls are instant.
- **Lazy 2D projection.** UMAP's first call costs ~25s of numba JIT. The MCP `load` tool skips it by default; `project_2d` is its own tool, called only when a scatter view is actually needed.
- **Stateful per-session, stateless per-call.** The MCP server holds one `Curator` for the lifetime of the client subprocess. Tools are pure functions over that state.

## Quickstart

### Prerequisites

- Python 3.10+
- [Ollama](https://ollama.com) running locally
- Pull the embedding model:
  ```bash
  ollama pull bge-m3
  ```

### CLI install + run

```bash
pip install -r requirements.txt
python curate.py --input data/sample.csv --text-column text
```

### MCP server install + run

```bash
cd mcp_server
python -m venv .venv && .venv\Scripts\activate     # Windows
pip install -e ".[viewer]"                          # add the [viewer] extra for Spotlight
curator-mcp                                         # speaks MCP over stdio
```

Then register it in your MCP client. For Claude Desktop (`claude_desktop_config.json`):

```jsonc
{
  "mcpServers": {
    "curator": {
      "command": "C:/path/to/whittle/mcp_server/.venv/Scripts/curator-mcp.exe"
    }
  }
}
```

Restart the client and ask it to `load` your dataset.

## MCP tools

The server exposes 11 tools, in the order a typical curation session uses them:

| Tool | What it does |
|------|--------------|
| `load` | Read CSV/JSONL, embed each row via Ollama (cached), optionally UMAP-project |
| `status` | Snapshot of what's currently loaded |
| `get_row` | Full content of one row by index |
| `search` | Top-k semantic neighbors for a free-form query |
| `find_outliers` | Most-isolated rows (low mean similarity to nearest neighbors) — removal candidates |
| `find_near_duplicates` | Pairs above a cosine threshold — dedup candidates |
| `mark` | Set `keep` / `flag` / `notes` on a list of row indices |
| `project_2d` | Run UMAP to add x/y coords (skipped by default at load) |
| `launch_viewer` | Open Spotlight in the browser, non-blocking |
| `close_viewer` | Stop the running Spotlight server |
| `save_kept` / `save_curated` | Write the molded subset, or the full annotated dataset, back to disk |

All tools are typed with Pydantic field descriptions, so the LLM client gets rich autocomplete and the right argument hints for free.

## How it was built — vibe coding the whole stack

This whole repo was built by *vibe coding* with [Claude Code](https://www.anthropic.com/claude-code) — a tight loop where the human describes intent, the agent writes the code, both run smoke tests, and the whole thing happens through conversation rather than IDE-driven keystrokes.

A few techniques that made it work on a project this size:

**Phase-gated smoke tests.** The MCP rebuild wasn't one big PR — it was four phases (load, inspect, mark/save, project/viewer), each with its own `smoke_phase{N}.py` that exercises every tool added in that phase end-to-end. Running the smoke after each phase caught regressions instantly and gave the agent a deterministic "is it done?" signal instead of vibes.

**Separate state from tools.** [`curator.py`](mcp_server/curator_mcp/curator.py) owns the in-memory dataset and pipeline; [`server.py`](mcp_server/curator_mcp/server.py) is just thin `@mcp.tool()` adapters with Pydantic-annotated arguments. The tools are trivial to read, and the agent could rewrite either layer without touching the other.

**Cache early, cache aggressively.** First implementation embedded everything every time. Adding [`cache.py`](mcp_server/curator_mcp/cache.py) (a tiny SHA256-keyed SQLite store) cut load-time from ~minutes to ~milliseconds on re-runs, which made the iteration loop fast enough to actually iterate.

**Lazy heavy deps.** `umap-learn` and `renumics-spotlight` are slow to import and slow to JIT. They're imported inside the methods that need them, and Spotlight is an optional `[viewer]` extra. The MCP server starts in well under a second.

**Treat the LLM client as a UI.** Spotlight Community Edition's table is read-only — that's a dead-end for a CLI workflow but a *feature* for an MCP workflow, because the editing UI is the LLM client itself. Reframing that constraint instead of fighting it removed an entire pile of code.

**Memory across sessions.** The agent maintained a small file-based memory of the user's preferences and the project's design decisions across multiple multi-hour sessions, so each "continue where we left off" picked up with full context instead of re-litigating choices.

## Roadmap

### Shipped

- [x] **Phase 1 — Load & embed.** CSV/JSONL ingestion, Ollama + OpenAI-compatible backends, SQLite embedding cache, `load` / `status` tools.
- [x] **Phase 2 — Inspect.** `get_row`, semantic `search`, `find_outliers`, `find_near_duplicates`.
- [x] **Phase 3 — Curate & save.** `mark` (keep/flag/notes), `save_kept`, `save_curated`.
- [x] **Phase 4 — Visual viewer.** `project_2d`, non-blocking `launch_viewer`, `close_viewer`.

### Next

- [ ] **Batch embedding for OpenAI-compatible backends** — Ollama is one-at-a-time, but `/v1/embeddings` accepts arrays. Easy ~5x speedup on LM Studio / vLLM.
- [ ] **Streaming progress events.** Long `load` calls should emit MCP progress notifications instead of going silent for minutes on first-time embedding.
- [ ] **Cluster labeling tool.** HDBSCAN over the embeddings, then ask the LLM to summarize each cluster — turns "find outliers" into "find the entire junk cluster".
- [ ] **Image / multimodal mode.** Same workflow with CLIP-style embeddings instead of text — Spotlight already renders thumbnails.
- [ ] **Diff view.** Compare two `.curated.csv` snapshots so you can see what changed across sessions.

### Maybe

- [ ] Web-hosted multiplayer mode (would break the local-first guarantee — only behind an explicit opt-in).
- [ ] Active-learning loop: pick the next 50 rows to label by uncertainty, not by hand.

## Project layout

```
whittle/
├── curate.py                    # CLI entry point
├── smoke_test.py                # CLI pipeline smoke test (no Spotlight launch)
├── requirements.txt             # CLI deps
├── data/
│   ├── sample.csv               # 18-row demo dataset
│   └── sample.csv.embcache.sqlite
└── mcp_server/
    ├── pyproject.toml           # MCP package + [viewer] extra
    ├── smoke_phase1.py … 4.py   # Phase-gated smoke tests
    └── curator_mcp/
        ├── server.py            # FastMCP tool definitions
        ├── curator.py           # Stateful pipeline (load, search, mark, save, project, viewer)
        ├── embeddings.py        # Ollama + OpenAI-compatible embedding clients
        └── cache.py             # SQLite embedding cache
```

---

<div align="center">

Built locally. Curated by conversation.

</div>
