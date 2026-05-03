"""Smoke test: validates load, embed, and UMAP without launching Spotlight."""
from pathlib import Path
import sys
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from curate import (
    load_dataset,
    embed_texts,
    project_2d,
    write_curated_template,
    apply_edits_and_save_kept,
    DEFAULT_EMBED_MODEL,
)

src = Path(__file__).parent / "data" / "sample.csv"
print(f"[1/4] Loading {src}")
df = load_dataset(src, text_column="text")
print(f"      rows={len(df)}, columns={list(df.columns)}")

# Embed only a subset for speed.
subset = df.head(5).copy()
print(f"[2/4] Embedding {len(subset)} rows via Ollama ({DEFAULT_EMBED_MODEL})")
embs = embed_texts(subset["text"].tolist(), model=DEFAULT_EMBED_MODEL)
print(f"      embeddings shape={embs.shape}, dtype={embs.dtype}")
assert embs.shape[1] > 1024 or embs.shape[1] == 1024, "expected >=1024-dim embeddings"

print(f"[3/4] UMAP -> 2D")
xy = project_2d(embs)
print(f"      coords shape={xy.shape}")
assert xy.shape == (len(subset), 2)

print(f"[4/4] write_curated_template + apply_edits_and_save_kept dry run")
subset["x"], subset["y"] = xy[:, 0], xy[:, 1]
subset["embedding"] = list(embs)
fake_src = src.with_name("smoke_sample.csv")
curated = write_curated_template(subset, fake_src)

# Simulate the user editing the .curated.csv to drop one row.
edited = pd.read_csv(curated) if curated.suffix == ".csv" else pd.read_json(curated, lines=True)
edited.loc[0, "keep"] = False
edited.to_csv(curated, index=False)

kept = apply_edits_and_save_kept(fake_src)
assert kept.exists()

print("OK - pipeline works end-to-end (Spotlight UI not launched in smoke test).")
