"""Quick health-check: verifies config, embedding backend, and dataset load."""
import json
import sys
from pathlib import Path

ROOT    = Path(__file__).resolve().parent
MCP_DIR = ROOT / "mcp_server"
sys.path.insert(0, str(MCP_DIR))

from curator_mcp.embeddings import EmbeddingConfig, DEFAULT_OLLAMA_URL
from curator_mcp.curator import Curator

config_path = MCP_DIR / "whittle.config.json"
if not config_path.exists():
    print("ERROR: whittle.config.json not found. Run install.py first.")
    sys.exit(1)

cfg = json.loads(config_path.read_text(encoding="utf-8"))
print(f"Config:   {config_path}")
print(f"Backend:  {cfg.get('embedding_backend', 'ollama')}")
print(f"URL:      {cfg.get('embedding_base_url', DEFAULT_OLLAMA_URL)}")
print(f"Model:    {cfg.get('embedding_model', 'bge-m3')}")
print(f"Prewarm:  {cfg.get('prewarm_umap', False)}")

al = cfg.get("auto_load")
if not (al and al.get("path")):
    print("\nNo auto_load configured - run install.py to set a dataset.")
    sys.exit(0)

print(f"Dataset:  {al['path']}")
print(f"Column:   {al.get('text_column', 'text')}")
print()

backend  = cfg.get("embedding_backend", "ollama")
base_url = cfg.get("embedding_base_url") or (DEFAULT_OLLAMA_URL if backend == "ollama" else "http://localhost:1234/v1")
ecfg = EmbeddingConfig(
    backend=backend,   # type: ignore[arg-type]
    base_url=base_url,
    model=cfg.get("embedding_model", "bge-m3"),
    api_key=cfg.get("embedding_api_key"),
)

print("Loading dataset (hits cache if already embedded)...")
try:
    c = Curator()
    r = c.load(path=al["path"], text_column=al.get("text_column", "text"), embedding_config=ecfg)
    c.close()
    print(f"  {r['rows']} rows  |  {r['embedding_dim']}-dim  |  {r['embeddings_cached_hits']} cache hits  |  {r['embeddings_freshly_computed']} fresh")
    print()
    print("OK - server is configured correctly.")
    print("LM Studio will start it automatically when you send a message.")
except Exception as exc:
    print(f"ERROR: {exc}")
    sys.exit(1)
