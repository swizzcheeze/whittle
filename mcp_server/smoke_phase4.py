"""
Phase 4 smoke test: project_2d + launch_viewer + close_viewer.

Skips actually opening a browser tab; just verifies the server starts on a port
and we can shut it down cleanly.
"""
from pathlib import Path
import sys
import time
import urllib.request

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

print("[1/3] project_2d (numba JIT cost on first call)")
t0 = time.time()
res = c.project_2d()
print(f"      {res}  (took {time.time()-t0:.1f}s)")
assert res["rows"] == 18

print("[2/3] launch_viewer (non-blocking)")
res = c.launch_viewer()
print(f"      {res}")
assert res["url"].startswith("http://")
# Probe the server is actually serving.
resp = urllib.request.urlopen(res["url"], timeout=10)
print(f"      probe HTTP {resp.status}")
assert resp.status == 200

print("[3/3] close_viewer")
res = c.close_viewer()
print(f"      {res}")
assert res["closed"] is True

# closing twice is safe
res = c.close_viewer()
assert res["closed"] is False

c.close()
print("OK - Phase 4 (project_2d + launch_viewer + close_viewer) works.")
