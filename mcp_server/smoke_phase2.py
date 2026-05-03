"""
Phase 2 smoke test: exercises the read-only inspection tools (get_row, search,
find_outliers, find_near_duplicates) directly on the Curator (skipping MCP transport).
"""
from pathlib import Path
import sys

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

print("[1/4] get_row(0)")
row = c.get_row(0)
print(f"      keys={list(row.keys())}")
print(f"      text[:60]={row['text'][:60]!r}")
assert row["row_index"] == 0
assert "embedding" not in row and "x" not in row and "y" not in row

print("[2/4] search('mitochondria photosynthesis cell biology', k=3)")
hits = c.search("mitochondria photosynthesis cell biology", k=3)
for h in hits:
    print(f"      idx={h['row_index']:>2}  score={h['score']:.3f}  text={h['text'][:55]!r}")
assert len(hits) == 3
# Top result should be a biology row given the query.
top_text = hits[0]["text"].lower()
assert any(kw in top_text for kw in ("mitochond", "photosynth", "ribosom", "dna", "biology"))

print("[3/4] find_outliers(top_k=4)")
outs = c.find_outliers(top_k=4, n_neighbors=3)
for o in outs:
    print(f"      idx={o['row_index']:>2}  outlier_score={o['outlier_score']:.3f}  text={o['text'][:55]!r}")
assert len(outs) == 4

print("[4/4] find_near_duplicates(threshold=0.7, top_k=5)")
# Threshold 0.7 should catch the noise rows (12, 16, 17, 18) which are near-duplicates of each other.
dups = c.find_near_duplicates(threshold=0.7, top_k=5)
for d in dups:
    print(f"      sim={d['similarity']:.3f}  a[idx={d['row_a']['row_index']}] vs b[idx={d['row_b']['row_index']}]")
assert len(dups) >= 1, "expected at least one near-duplicate pair among the 4 noise rows"

c.close()
print("OK - Phase 2 (get_row + search + find_outliers + find_near_duplicates) works.")
