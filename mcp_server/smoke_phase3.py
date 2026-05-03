"""
Phase 3 smoke test: mark + save_kept + save_curated.
"""
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from curator_mcp.curator import Curator
from curator_mcp.embeddings import EmbeddingConfig


sample = (ROOT.parent / "data" / "sample.csv").resolve()

c = Curator()
c.load(
    path=str(sample),
    text_column="text",
    embedding_config=EmbeddingConfig(model="bge-m3"),
    reduce_to_2d=False,
)

print("[1/4] mark drop on 4 noise rows (idx 11,15,16,17)")
res = c.mark(row_indices=[11, 15, 16, 17], keep=False, notes="lorem-ipsum noise")
print(f"      {res}")
assert res["kept_count"] == 18 - 4
assert res["noted_count"] == 4

print("[2/4] mark flag on a few spam rows (idx 6, 7, 12)")
res = c.mark(row_indices=[6, 7, 12], flag=True)
print(f"      {res}")
assert res["flagged_count"] == 3

print("[3/4] save_kept to a temp file")
out_path = sample.with_name("smoke_kept.csv")
res = c.save_kept(output_path=str(out_path))
print(f"      {res}")
kept = pd.read_csv(out_path)
print(f"      kept rows={len(kept)}, columns={list(kept.columns)}")
assert len(kept) == 14
assert "keep" not in kept.columns and "embedding" not in kept.columns
# flag/notes should still be present so reviewers can see why something was kept-but-flagged.
assert "flag" in kept.columns and "notes" in kept.columns

print("[4/4] save_curated to a temp file (full annotated)")
out_full = sample.with_name("smoke_curated.csv")
res = c.save_curated(output_path=str(out_full))
print(f"      {res}")
full = pd.read_csv(out_full)
assert len(full) == 18
assert "keep" in full.columns

# clean up smoke artifacts
out_path.unlink()
out_full.unlink()

c.close()
print("OK - Phase 3 (mark + save_kept + save_curated) works.")
