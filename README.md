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

> **Why this exists.** The bottleneck in modern ML is no longer model architecture — it's data quality. But every "data-centric AI" tool ships your text to a SaaS dashboard. This project keeps the entire loop on your laptop: embeddings via Ollama or LM Studio, dimensionality reduction via UMAP, an in-browser scatter view via Spotlight, and an MCP server so any LLM client (Claude Desktop, Claude Code, LM Studio, Continue, Goose) can drive the whole curation workflow in plain English.

## Contents

- [What it is](#what-it-is)
- [The two ways to use it](#the-two-ways-to-use-it)
- [Architecture](#architecture)
- [Quickstart — one command](#quickstart--one-command)
- [Connecting to LM Studio](#connecting-to-lm-studio)
- [MCP tools](#mcp-tools)
- [Configuration reference](#configuration-reference)
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

- **No SaaS, no API keys.** Ollama or LM Studio for embeddings — your text never leaves the machine.
- **Caching is non-negotiable.** Every embedding is keyed by `SHA256(model || text)` and stored in a SQLite file beside the dataset. Re-runs and "load again" calls are instant.
- **Lazy 2D projection.** UMAP's first call costs ~25s of numba JIT. The MCP `load` tool skips it by default; `project_2d` is its own tool, called only when a scatter view is actually needed.
- **Stateful per-session, stateless per-call.** The MCP server holds one `Curator` for the lifetime of the client subprocess. Tools are pure functions over that state.
- **Resilient to client restarts.** Some MCP clients (LM Studio) restart the server subprocess between tool calls. `auto_load` in the config reloads your dataset on every server start so tools are never left stateless.

## Quickstart — one command

```bash
python install.py
```

The interactive installer handles everything:

| Step | What it asks |
|------|-------------|
| Mode | `mcp` / `cli` / `both` |
| Viewer | Install Spotlight browser UI? |
| Backend | Ollama · LM Studio · OpenAI-compatible |
| URL + model | Where to send embedding requests |
| Auto-load | Load a dataset automatically on every server start |
| Pre-warm UMAP | Compile numba in the background at startup (prevents `project_2d` timeouts) |

At the end it prints the exact JSON snippet to paste into your MCP client config and restarts cleanly. No manual venv steps needed.

### Prerequisites

- Python 3.10+
- One of: [Ollama](https://ollama.com) with `ollama pull bge-m3`, or LM Studio with an embedding model loaded

### After setup — registering the server

The installer prints the right snippet for your client. For reference:

**Claude Desktop** (`%APPDATA%\Claude\claude_desktop_config.json`):
```jsonc
{
  "mcpServers": {
    "whittle": {
      "command": "C:/path/to/whittle/mcp_server/.venv/Scripts/curator-mcp.exe"
    }
  }
}
```

**LM Studio** (Settings → Local Server → MCP Servers):
```json
{
  "mcpServers": {
    "whittle": {
      "command": "C:\\path\\to\\whittle\\mcp_server\\.venv\\Scripts\\curator-mcp.exe"
    }
  }
}
```

**Claude Code** (`.claude/settings.json` in your project):
```json
{
  "mcpServers": {
    "whittle": {
      "command": "C:/path/to/whittle/mcp_server/.venv/Scripts/curator-mcp.exe",
      "type": "stdio"
    }
  }
}
```

Restart the client after saving, then ask it to `load` your dataset.

## Connecting to LM Studio

LM Studio works as both the **LLM client** (driving the tools) and optionally the **embedding backend** (replacing Ollama). A few known quirks and how Whittle handles them:

| Issue | Root cause | Fix |
|-------|-----------|-----|
| "No dataset loaded" after a successful `load` | LM Studio restarts the MCP server subprocess between tool calls | Enable `auto_load` in the installer — the server reloads the dataset on every start |
| `project_2d` times out / "operation aborted" | numba JIT takes ~25s on first call, LM Studio's request timeout is shorter | Enable `prewarm_umap` in the installer — JIT runs in the background at startup |
| Embeddings via LM Studio instead of Ollama | LM Studio exposes an OpenAI-compatible `/v1/embeddings` endpoint | Choose `lmstudio` backend in the installer; enter the model identifier from LM Studio's UI |

**Recommended LM Studio setup** (run `python install.py` and answer):
```
Mode       → mcp
Viewer     → yes
Backend    → lmstudio
URL        → http://localhost:1234/v1
Model      → (copy from LM Studio's loaded embedding model)
Auto-load  → yes  (path to your dataset)
Pre-warm   → yes
```

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

## Configuration reference

`install.py` writes `mcp_server/whittle.config.json`. You can also edit it by hand — see [`whittle.config.example.json`](mcp_server/whittle.config.example.json) for the full schema.

```jsonc
{
  // "ollama" or "openai" (LM Studio and any OpenAI-compatible server use "openai")
  "embedding_backend": "ollama",

  // URL of the embedding server
  "embedding_base_url": "http://localhost:11434",

  // Model identifier — must match what's loaded in your embedding server
  "embedding_model": "bge-m3",

  // Pre-warm numba JIT at startup so project_2d never times out (recommended: true)
  "prewarm_umap": true,

  // OpenAI/LM Studio only: texts sent per /v1/embeddings call (default: 64)
  // Raise to 128-256 if your server has headroom; lower if you hit 413/timeout
  "embedding_batch_size": 64,

  // Optional: load a dataset automatically on every server start.
  // Prevents "No dataset loaded" errors when the MCP client restarts the server.
  "auto_load": {
    "path": "C:/absolute/path/to/your/dataset.csv",
    "text_column": "text"
  },

  // Optional: only needed for some OpenAI-compatible backends
  "embedding_api_key": "sk-..."
}
```

The server searches for `whittle.config.json` at `mcp_server/whittle.config.json`. If the file doesn't exist the server starts with Ollama + bge-m3 defaults and no auto-load.

## How it was built — vibe coding the whole stack

This whole repo was built by *vibe coding* with [Claude Code](https://www.anthropic.com/claude-code) 

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
- [x] **Interactive installer.** `install.py` wizard — venv, deps, backend choice (Ollama/LM Studio/OpenAI), auto-load, UMAP pre-warm, MCP config snippet output.
- [x] **LM Studio compatibility.** Auto-load on startup fixes state-loss between tool calls; UMAP pre-warm prevents `project_2d` timeouts.
- [x] **Batch embedding for OpenAI-compatible backends.** `/v1/embeddings` accepts arrays; `embedding_batch_size` (default 64) controls chunk size. Each chunk is cached on completion so partial progress survives an interrupted run. Ollama stays one-at-a-time (no batch API there).

### Next

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
├── install.py                       # interactive setup wizard  ← start here
├── curate.py                        # CLI entry point
├── smoke_test.py                    # CLI pipeline smoke test
├── requirements.txt                 # CLI deps
├── data/
│   └── sample.csv                   # 18-row demo dataset
└── mcp_server/
    ├── pyproject.toml               # MCP package + [viewer] extra
    ├── whittle.config.example.json  # config schema reference
    ├── whittle.config.json          # generated by install.py (gitignored)
    ├── smoke_phase1.py … 4.py       # phase-gated smoke tests
    └── curator_mcp/
        ├── server.py                # FastMCP tools + startup (config, auto-load, prewarm)
        ├── curator.py               # stateful pipeline (load, search, mark, save, project, viewer)
        ├── embeddings.py            # Ollama + OpenAI-compatible embedding clients
        └── cache.py                 # SQLite embedding cache
```

---

<div align="center">

Built locally. Curated by conversation.

</div>
