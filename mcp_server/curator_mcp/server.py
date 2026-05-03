"""
FastMCP server. Exposes curation tools over MCP/stdio so any MCP-compatible client
(Claude Desktop, Claude Code, Continue, Goose, LM Studio, etc.) can drive the workflow.
"""
from __future__ import annotations

import json
import sys
import threading
from pathlib import Path
from typing import Annotated

from mcp.server.fastmcp import FastMCP
from pydantic import Field

from .curator import Curator
from .embeddings import EmbeddingConfig, DEFAULT_OLLAMA_URL


# Single per-process Curator. MCP servers run as a persistent subprocess of the client,
# so this lives for the duration of the client session.
_curator = Curator()

# Startup config loaded from whittle.config.json (written by install.py).
_startup_cfg: dict = {}

mcp = FastMCP(
    "curator-mcp",
    instructions=(
        "Tools for curating local text datasets with embeddings. "
        "Always call `load` first; subsequent tools operate on the loaded dataset. "
        "Use `status` to check what's currently loaded."
    ),
)


def _read_startup_config() -> dict:
    """Look for whittle.config.json next to the mcp_server directory."""
    here = Path(__file__).resolve().parent
    for candidate in (here.parent / "whittle.config.json", here / "whittle.config.json"):
        if candidate.exists():
            try:
                return json.loads(candidate.read_text(encoding="utf-8"))
            except Exception:
                pass
    return {}


def _cfg_embedding() -> EmbeddingConfig:
    """Build an EmbeddingConfig from startup config (or sensible defaults)."""
    backend  = _startup_cfg.get("embedding_backend", "ollama")
    base_url = _startup_cfg.get("embedding_base_url")
    model    = _startup_cfg.get("embedding_model", "bge-m3")
    api_key  = _startup_cfg.get("embedding_api_key")
    if not base_url:
        base_url = DEFAULT_OLLAMA_URL if backend == "ollama" else "http://localhost:1234/v1"
    return EmbeddingConfig(
        backend=backend,   # type: ignore[arg-type]
        base_url=base_url,
        model=model,
        api_key=api_key,
    )


def _prewarm_umap() -> None:
    """Run a tiny UMAP fit in a daemon thread to trigger numba JIT compilation.
    This front-loads the ~25s first-call cost so project_2d responds in ~1s."""
    try:
        import numpy as np
        import umap as _umap
        dummy = np.random.default_rng(0).random((12, 64)).astype("float32")
        _umap.UMAP(n_components=2, n_neighbors=5, random_state=0).fit_transform(dummy)
        print("[whittle] UMAP pre-warm complete.", file=sys.stderr, flush=True)
    except Exception as exc:
        print(f"[whittle] UMAP pre-warm skipped: {exc}", file=sys.stderr, flush=True)


@mcp.tool()
def load(
    path: Annotated[str, Field(description="Absolute or relative path to a .csv or .jsonl dataset.")],
    text_column: Annotated[str, Field(description="Name of the column whose text should be embedded.")] = "text",
    id_column: Annotated[str | None, Field(description="Optional name of a stable id column. If omitted, row index is used.")] = None,
    backend: Annotated[str | None, Field(description="Embedding backend: 'ollama' or 'openai' (any OpenAI-compatible /v1/embeddings server). Defaults to value from whittle.config.json.")] = None,
    model: Annotated[str | None, Field(description="Embedding model name. Defaults to value from whittle.config.json, or 'bge-m3'.")] = None,
    base_url: Annotated[str | None, Field(description="Override embedding server URL. Defaults to value from whittle.config.json.")] = None,
    api_key: Annotated[str | None, Field(description="Optional API key for OpenAI-compatible backends.")] = None,
    reduce_to_2d: Annotated[bool, Field(description="Run UMAP to add 2D x/y coordinates. Default false; use the separate project_2d tool when you actually need a scatter view.")] = False,
) -> dict:
    """Load a CSV or JSONL dataset and embed each row.

    Embeddings are cached to a sibling SQLite file so re-loads are instant.
    Returns a summary including row count, embedding dimensionality, and
    cache hit rate. By default UMAP is NOT run — call project_2d when you want it.

    Backend/model/url default to values set in whittle.config.json (written by install.py).
    """
    # Merge call-time overrides on top of startup config defaults.
    base_cfg = _cfg_embedding()
    resolved_backend  = backend  or base_cfg.backend
    resolved_model    = model    or base_cfg.model
    resolved_base_url = base_url or base_cfg.base_url
    resolved_api_key  = api_key  or base_cfg.api_key

    if resolved_backend not in ("ollama", "openai"):
        raise ValueError(f"backend must be 'ollama' or 'openai', got {resolved_backend!r}")

    cfg = EmbeddingConfig(
        backend=resolved_backend,   # type: ignore[arg-type]
        base_url=resolved_base_url,
        model=resolved_model,
        api_key=resolved_api_key,
    )
    return _curator.load(
        path=path,
        text_column=text_column,
        id_column=id_column,
        embedding_config=cfg,
        reduce_to_2d=reduce_to_2d,
    )


@mcp.tool()
def status() -> dict:
    """Return a snapshot of the currently loaded dataset (or {loaded: false} if none)."""
    return _curator.status()


@mcp.tool()
def get_row(
    row_index: Annotated[int, Field(description="Zero-based row index in the loaded dataset.")],
) -> dict:
    """Return the full content of one row (all columns, JSON-serialized).

    Use this after `search` / `find_outliers` / `find_near_duplicates` returned a
    truncated snippet and you need the complete text.
    """
    return _curator.get_row(row_index)


@mcp.tool()
def search(
    query: Annotated[str, Field(description="Free-form text query. Will be embedded with the same model used at load time.")],
    k: Annotated[int, Field(description="Number of nearest rows to return.", ge=1, le=200)] = 5,
) -> list[dict]:
    """Semantic search: return the top-k rows most similar to the query.

    Each result includes row_index, a short snippet, keep/flag state, and a cosine
    similarity score (higher = more similar; 1.0 = identical direction).
    """
    return _curator.search(query=query, k=k)


@mcp.tool()
def find_outliers(
    top_k: Annotated[int, Field(description="How many of the most-isolated rows to return.", ge=1, le=500)] = 10,
    n_neighbors: Annotated[int, Field(description="Number of nearest neighbors used to score isolation.", ge=1, le=100)] = 5,
) -> list[dict]:
    """Return rows that sit far from any cluster — candidates for removal/flagging.

    Score = 1 - mean cosine similarity to the row's `n_neighbors` nearest peers.
    Higher score = more isolated.
    """
    return _curator.find_outliers(top_k=top_k, n_neighbors=n_neighbors)


@mcp.tool()
def find_near_duplicates(
    threshold: Annotated[float, Field(description="Cosine similarity cutoff. Pairs with sim >= threshold are returned.", ge=0.0, le=1.0)] = 0.95,
    top_k: Annotated[int, Field(description="Maximum number of pairs to return.", ge=1, le=1000)] = 50,
) -> list[dict]:
    """Return pairs of rows that look like near-duplicates (cosine sim above threshold).

    Useful for dedup. Each pair includes `row_a`, `row_b`, and `similarity`.
    """
    return _curator.find_near_duplicates(threshold=threshold, top_k=top_k)


@mcp.tool()
def mark(
    row_indices: Annotated[list[int], Field(description="Zero-based row indices to update. Pass a single index as a one-element list.")],
    keep: Annotated[bool | None, Field(description="If set, write this value to the 'keep' column for the listed rows.")] = None,
    flag: Annotated[bool | None, Field(description="If set, write this value to the 'flag' column for the listed rows.")] = None,
    notes: Annotated[str | None, Field(description="If set, write this string to the 'notes' column for the listed rows.")] = None,
) -> dict:
    """Apply curation flags to rows. At least one of keep/flag/notes must be supplied.

    Examples (in plain English the LLM will translate):
      - "drop rows 12, 16, 17, 18"      -> mark([12,16,17,18], keep=False)
      - "flag the spam-looking row 7"   -> mark([7], flag=True, notes="spam suspect")
      - "restore row 5"                 -> mark([5], keep=True)
    """
    return _curator.mark(row_indices=row_indices, keep=keep, flag=flag, notes=notes)


@mcp.tool()
def save_kept(
    output_path: Annotated[str | None, Field(description="Optional output path. Defaults to <source>.kept.<ext> next to the source file.")] = None,
) -> dict:
    """Write the molded subset (rows where keep == True) to disk. This is the 'final product' of curation."""
    return _curator.save_kept(output_path=output_path)


@mcp.tool()
def save_curated(
    output_path: Annotated[str | None, Field(description="Optional output path. Defaults to <source>.curated.<ext> next to the source file.")] = None,
) -> dict:
    """Write the full annotated dataset (with keep/flag/notes columns) for review or later re-loading."""
    return _curator.save_curated(output_path=output_path)


@mcp.tool()
def project_2d(
    seed: Annotated[int, Field(description="Random seed for the UMAP projection (for reproducibility).")] = 42,
) -> dict:
    """Run UMAP on the embeddings to produce 2D x/y coordinates.

    First call in a fresh process is slow (~25s) due to numba JIT compilation;
    subsequent calls are fast. Required before launch_viewer if you want a meaningful
    scatter plot.
    """
    return _curator.project_2d(seed=seed)


@mcp.tool()
def launch_viewer() -> dict:
    """Open Spotlight in the user's browser as a non-blocking visual viewer.

    Auto-runs project_2d if not already done. Returns the URL of the running server
    so the caller can share/open it. The viewer is read-only (Spotlight CE limitation);
    use the `mark` tool to actually edit rows.
    """
    return _curator.launch_viewer()


@mcp.tool()
def close_viewer() -> dict:
    """Stop the running Spotlight viewer (if any). Safe to call when no viewer is open."""
    return _curator.close_viewer()


def main() -> None:
    """Console entry point. Runs the server over stdio."""
    global _startup_cfg
    _startup_cfg = _read_startup_config()

    # Pre-warm numba JIT in the background so project_2d responds in ~1s instead of ~25s.
    if _startup_cfg.get("prewarm_umap", False):
        threading.Thread(target=_prewarm_umap, daemon=True, name="umap-prewarm").start()

    # Auto-load: if the config names a dataset, load it immediately.
    # This ensures tools work even if the MCP client restarts the server process mid-session.
    if (al := _startup_cfg.get("auto_load")) and isinstance(al, dict) and al.get("path"):
        try:
            print(f"[whittle] auto-loading {al['path']} ...", file=sys.stderr, flush=True)
            _curator.load(
                path=al["path"],
                text_column=al.get("text_column", "text"),
                embedding_config=_cfg_embedding(),
            )
            print("[whittle] auto-load complete.", file=sys.stderr, flush=True)
        except Exception as exc:
            print(f"[whittle] auto-load failed: {exc}", file=sys.stderr, flush=True)

    mcp.run()


if __name__ == "__main__":
    main()
