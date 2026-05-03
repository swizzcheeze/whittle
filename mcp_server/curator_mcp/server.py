"""
FastMCP server. Exposes curation tools over MCP/stdio so any MCP-compatible client
(Claude Desktop, Claude Code, Continue, Goose, etc.) can drive the workflow.

Phase 1 tools:
  - load:    open a dataset, embed it, optionally project to 2D
  - status:  inspect what's currently loaded
"""
from __future__ import annotations

from typing import Annotated

from mcp.server.fastmcp import FastMCP
from pydantic import Field

from .curator import Curator
from .embeddings import EmbeddingConfig, DEFAULT_OLLAMA_URL


# Single per-process Curator. MCP servers run as a persistent subprocess of the client,
# so this lives for the duration of the client session.
_curator = Curator()

mcp = FastMCP(
    "curator-mcp",
    instructions=(
        "Tools for curating local text datasets with embeddings. "
        "Always call `load` first; subsequent tools operate on the loaded dataset. "
        "Use `status` to check what's currently loaded."
    ),
)


@mcp.tool()
def load(
    path: Annotated[str, Field(description="Absolute or relative path to a .csv or .jsonl dataset.")],
    text_column: Annotated[str, Field(description="Name of the column whose text should be embedded.")] = "text",
    id_column: Annotated[str | None, Field(description="Optional name of a stable id column. If omitted, row index is used.")] = None,
    backend: Annotated[str, Field(description="Embedding backend: 'ollama' (default) or 'openai' (any OpenAI-compatible /v1/embeddings server).")] = "ollama",
    model: Annotated[str, Field(description="Embedding model name. Defaults to 'bge-m3' (Ollama).")] = "bge-m3",
    base_url: Annotated[str | None, Field(description="Override embedding server URL. Defaults to http://localhost:11434 for Ollama.")] = None,
    api_key: Annotated[str | None, Field(description="Optional API key for OpenAI-compatible backends.")] = None,
    reduce_to_2d: Annotated[bool, Field(description="Run UMAP to add 2D x/y coordinates. Default false; use the separate project_2d tool when you actually need a scatter view.")] = False,
) -> dict:
    """Load a CSV or JSONL dataset and embed each row.

    Embeddings are cached to a sibling SQLite file so re-loads are instant.
    Returns a summary including row count, embedding dimensionality, and
    cache hit rate. By default UMAP is NOT run — call project_2d when you want it.
    """
    if backend not in ("ollama", "openai"):
        raise ValueError(f"backend must be 'ollama' or 'openai', got {backend!r}")
    cfg = EmbeddingConfig(
        backend=backend,                                       # type: ignore[arg-type]
        base_url=base_url or (DEFAULT_OLLAMA_URL if backend == "ollama" else "http://localhost:1234/v1"),
        model=model,
        api_key=api_key,
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
    mcp.run()


if __name__ == "__main__":
    main()
