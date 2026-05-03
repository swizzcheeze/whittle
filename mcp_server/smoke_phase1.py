"""
Phase 1 smoke test — validates the Curator core (load + status + cache) without
touching the MCP transport layer. Run this from the venv:

    .venv/Scripts/python.exe smoke_phase1.py
"""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from curator_mcp.curator import Curator
from curator_mcp.embeddings import EmbeddingConfig


sample = (ROOT.parent / "data" / "sample.csv").resolve()
assert sample.exists(), f"sample dataset missing: {sample}"

# Make the test reproducible: nuke any prior cache so the first load is truly cold.
cache_path = sample.with_suffix(sample.suffix + ".embcache.sqlite")
if cache_path.exists():
    cache_path.unlink()

c = Curator()
print("[1/3] First load (cold cache, hits Ollama)")
summary = c.load(
    path=str(sample),
    text_column="text",
    embedding_config=EmbeddingConfig(model="bge-m3"),
    reduce_to_2d=False,
)
print(f"      rows={summary['rows']}, dim={summary['embedding_dim']}, "
      f"hits={summary['embeddings_cached_hits']}, fresh={summary['embeddings_freshly_computed']}")
assert summary["rows"] > 0
assert summary["embedding_dim"] == 1024
assert summary["embeddings_cached_hits"] == 0       # first run: empty cache
assert summary["embeddings_freshly_computed"] == summary["rows"]

print("[2/3] Status snapshot")
st = c.status()
print(f"      loaded={st['loaded']}, columns={st['columns'][:6]}..., kept={st['kept_count']}")
assert st["loaded"] is True
assert "keep" in st["columns"] and "flag" in st["columns"] and "notes" in st["columns"]
assert st["kept_count"] == st["rows"]               # all rows start as keep=true
assert st["reduced_to_2d"] is False
assert st["cache"]["rows"] >= summary["rows"]

print("[3/3] Second load (warm cache; should be 100% hits)")
c2 = Curator()
summary2 = c2.load(
    path=str(sample),
    text_column="text",
    embedding_config=EmbeddingConfig(model="bge-m3"),
    reduce_to_2d=False,
)
print(f"      hits={summary2['embeddings_cached_hits']}, fresh={summary2['embeddings_freshly_computed']}")
assert summary2["embeddings_cached_hits"] == summary2["rows"]
assert summary2["embeddings_freshly_computed"] == 0

c.close()
c2.close()
print("OK - Phase 1 (load + status + cache) works end-to-end.")
