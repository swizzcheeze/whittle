"""
Pre-embed a dataset so LM Studio (or any MCP client) can load it instantly.

Usage:
    python embed.py path/to/dataset.csv
    python embed.py path/to/dataset.csv --column title
    python embed.py path/to/dataset.csv --column body --batch-size 128

Run this once before connecting LM Studio. All embeddings are cached in a
sibling .embcache.sqlite file, so subsequent loads return immediately from
cache rather than calling the embedding server.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT    = Path(__file__).resolve().parent
MCP_DIR = ROOT / "mcp_server"
sys.path.insert(0, str(MCP_DIR))

from curator_mcp.embeddings import EmbeddingConfig, DEFAULT_OLLAMA_URL
from curator_mcp.curator import Curator


def load_config() -> dict:
    p = MCP_DIR / "whittle.config.json"
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}


def main() -> None:
    cfg = load_config()

    parser = argparse.ArgumentParser(description="Pre-embed a dataset for Whittle.")
    parser.add_argument("path", nargs="?", help="Path to .csv or .jsonl dataset")
    parser.add_argument("--column", "-c", default=None, help="Text column to embed (auto-detected if omitted)")
    parser.add_argument("--batch-size", type=int, default=None, help="Texts per /v1/embeddings call (OpenAI/LM Studio only)")
    args = parser.parse_args()

    # Resolve dataset path: CLI arg > auto_load in config > error
    path = args.path
    text_column = args.column

    if not path:
        al = cfg.get("auto_load")
        if al and al.get("path"):
            path = al["path"]
            if not text_column:
                text_column = al.get("text_column", "text")
            print(f"Using auto_load path from config: {path}")
        else:
            print("Usage: python embed.py <path/to/dataset.csv> [--column <col>]")
            sys.exit(1)

    path = str(Path(path).expanduser().resolve())
    text_column = text_column or "text"

    # Build embedding config from whittle.config.json
    backend    = cfg.get("embedding_backend", "ollama")
    base_url   = cfg.get("embedding_base_url") or (DEFAULT_OLLAMA_URL if backend == "ollama" else "http://localhost:1234/v1")
    model      = cfg.get("embedding_model", "mxbai-embed-large")
    api_key    = cfg.get("embedding_api_key")
    batch_size = args.batch_size or cfg.get("embedding_batch_size")

    kwargs: dict = dict(
        backend=backend, base_url=base_url, model=model, api_key=api_key,
        timeout_s=300.0,  # generous timeout so a cold model load never fails
    )
    if batch_size is not None:
        kwargs["batch_size"] = int(batch_size)
    ecfg = EmbeddingConfig(**kwargs)  # type: ignore[arg-type]

    print(f"\nDataset : {path}")
    print(f"Column  : {text_column}")
    print(f"Backend : {backend}  |  model: {model}  |  url: {base_url}")
    if backend == "openai":
        print(f"Batch   : {ecfg.batch_size} texts per call")
    print()

    # Progress reporter: prints a line every ~5% or every 60s, whichever comes first.
    t_start = time.time()
    t_last_print = [t_start]
    last_pct = [-1]

    def on_progress(done: int, total: int) -> None:
        pct = int(done / total * 100) if total else 100
        now = time.time()
        elapsed = now - t_start
        # Print on first call, every 5% milestone, or every 60s
        if pct == last_pct[0] and (now - t_last_print[0]) < 60:
            return
        last_pct[0] = pct
        t_last_print[0] = now
        rate = done / elapsed if elapsed > 1 else 0
        eta_s = (total - done) / rate if rate > 0 else 0
        eta_str = _fmt_duration(eta_s) if eta_s > 0 else "..."
        bar = "#" * (pct // 5) + "-" * (20 - pct // 5)
        print(f"\r  [{bar}] {pct:3d}%  {done:,}/{total:,}  eta {eta_str}   ", end="", flush=True)

    c = Curator()
    try:
        r = c.load(
            path=path,
            text_column=text_column,
            embedding_config=ecfg,
            on_progress=on_progress,
        )
    except Exception as exc:
        print(f"\nERROR: {exc}")
        sys.exit(1)
    finally:
        c.close()

    elapsed = time.time() - t_start
    fresh   = r["embeddings_freshly_computed"]
    hits    = r["embeddings_cached_hits"]
    total   = r["rows"]

    print(f"\r  {'#'*20} 100%  {total:,}/{total:,}  done{' '*20}")
    print(f"\nFinished in {_fmt_duration(elapsed)}")
    print(f"  {total:,} rows  |  {r['embedding_dim']}-dim")
    print(f"  {fresh:,} freshly embedded  |  {hits:,} from cache")
    print()

    if fresh == 0:
        print("All embeddings were already cached - load will be instant.")
    else:
        col = r.get("text_column", text_column)
        print("Cache populated. LM Studio will now load this dataset in ~1s.")
        if col != text_column:
            print(f"  (Note: column auto-detected as '{col}')")
        print(f"\n  Add this to mcp_server/whittle.config.json to auto-load on startup:")
        print(f'    "auto_load": {{"path": "{path}", "text_column": "{col}"}}')
    print()


def _fmt_duration(seconds: float) -> str:
    seconds = int(seconds)
    if seconds < 60:
        return f"{seconds}s"
    m, s = divmod(seconds, 60)
    if m < 60:
        return f"{m}m{s:02d}s"
    h, m = divmod(m, 60)
    return f"{h}h{m:02d}m"


if __name__ == "__main__":
    main()
