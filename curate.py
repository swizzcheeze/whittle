"""
curate.py — Visual data curation with Spotlight + local Ollama embeddings.

Workflow:
    1. Load a CSV or JSONL file containing a text column.
    2. Embed each row locally via Ollama (default: mxbai-embed-large, 1024-dim).
    3. Reduce to 2D with UMAP for spatial browsing.
    4. Open Spotlight — click points, inspect raw text, tag rows to keep/drop.
    5. Save the curated ("molded") dataset back to disk.

Install:
    pip install pandas ollama renumics-spotlight umap-learn scikit-learn

Prereq:
    Ollama running locally with the embedding model pulled:
        ollama pull bge-m3

Usage:
    python curate.py --input data/sample.csv --text-column text
    python curate.py --input data/sample.jsonl --text-column content --model mxbai-embed-large
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import ollama
import umap
from renumics import spotlight
from renumics.spotlight import layout as sl


DEFAULT_EMBED_MODEL = "bge-m3"              # 1024-dim multilingual local embedding model
CURATED_SUFFIX = ".curated"                 # full annotated copy
KEPT_SUFFIX = ".kept"                       # only rows the user kept


# ---------- 1. Loading ----------

def load_dataset(path: Path, text_column: str) -> pd.DataFrame:
    """Load CSV or JSONL into a DataFrame and validate the text column."""
    suffix = path.suffix.lower()
    if suffix == ".csv":
        df = pd.read_csv(path)
    elif suffix in {".jsonl", ".ndjson"}:
        df = pd.read_json(path, lines=True)
    elif suffix == ".json":
        df = pd.read_json(path)
    else:
        raise ValueError(f"Unsupported file type: {suffix}. Use .csv or .jsonl")

    if text_column not in df.columns:
        raise KeyError(
            f"Column '{text_column}' not found. Available columns: {list(df.columns)}"
        )

    # Embeddings need real content — drop rows with empty/null text.
    df = df.dropna(subset=[text_column]).reset_index(drop=True)
    df[text_column] = df[text_column].astype(str)

    # Curation columns the user edits inside Spotlight.
    if "keep" not in df.columns:
        df["keep"] = True       # toggle off rows to drop
    if "flag" not in df.columns:
        df["flag"] = False      # mark rows for follow-up
    if "notes" not in df.columns:
        df["notes"] = ""        # free-form annotation

    return df


# ---------- 2. Embedding ----------

def embed_texts(texts: list[str], model: str) -> np.ndarray:
    """Embed each text via the local Ollama daemon. Returns an (N, D) float32 array."""
    vectors: list[list[float]] = []
    total = len(texts)
    for i, text in enumerate(texts, start=1):
        # Periodic progress to stderr — CPU embedding can be slow.
        if i == 1 or i % 25 == 0 or i == total:
            print(f"  embedding {i}/{total}", file=sys.stderr)
        resp = ollama.embeddings(model=model, prompt=text)
        vectors.append(resp["embedding"])
    return np.asarray(vectors, dtype=np.float32)


# ---------- 3. Dimensionality reduction ----------

def project_2d(embeddings: np.ndarray, seed: int = 42) -> np.ndarray:
    """UMAP projection of high-dim embeddings (>1024) to 2D coordinates."""
    n_neighbors = min(15, max(2, len(embeddings) - 1))
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=0.1,
        metric="cosine",
        random_state=seed,
    )
    return reducer.fit_transform(embeddings)


# ---------- 4. Save curated output ----------
#
# Spotlight Community Edition's table is read-only — you can't toggle
# `keep`/`flag` in the browser. The workflow instead:
#   1. The script writes <stem>.curated.<ext> BEFORE opening Spotlight
#      (all rows start with keep=True).
#   2. You explore in Spotlight (read-only viewer) and, in any text editor
#      / Excel / VSCode, edit `keep` (and optionally `flag`/`notes`) in
#      that .curated file.
#   3. You close the Spotlight tab.
#   4. The script re-reads the (possibly edited) .curated file and writes
#      <stem>.kept.<ext> with only the rows where keep == True.

def _curated_path(source_path: Path) -> Path:
    return source_path.with_suffix(CURATED_SUFFIX + source_path.suffix.lower())


def _kept_path(source_path: Path) -> Path:
    return source_path.with_suffix(KEPT_SUFFIX + source_path.suffix.lower())


def write_curated_template(df: pd.DataFrame, source_path: Path) -> Path:
    """Write the editable annotated copy. Strips embedding/x/y for editor-friendliness."""
    out = _curated_path(source_path)

    # Strip transient columns the user shouldn't have to scroll past.
    drop_cols = [c for c in ("embedding", "x", "y") if c in df.columns]
    annotated = df.drop(columns=drop_cols)

    if out.suffix == ".csv":
        annotated.to_csv(out, index=False)
    else:
        annotated.to_json(out, orient="records", lines=True, force_ascii=False)

    print(f"Wrote editable curated file -> {out}")
    print("  Edit the 'keep' column in this file (true/false). Save when done.")
    return out


def _coerce_bool(value) -> bool:
    """Accept true/false, True/False, 1/0, yes/no — anything sane an editor produces."""
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    s = str(value).strip().lower()
    return s in {"true", "1", "yes", "y", "t", "keep"}


def apply_edits_and_save_kept(source_path: Path) -> Path:
    """Read the (possibly user-edited) curated file and write the kept-only subset."""
    curated = _curated_path(source_path)
    kept_out = _kept_path(source_path)

    if curated.suffix == ".csv":
        df = pd.read_csv(curated)
    else:
        df = pd.read_json(curated, lines=True)

    df["keep"] = df["keep"].apply(_coerce_bool)
    kept = df[df["keep"]].drop(columns=["keep"])

    if kept_out.suffix == ".csv":
        kept.to_csv(kept_out, index=False)
    else:
        kept.to_json(kept_out, orient="records", lines=True, force_ascii=False)

    print(f"Saved kept-only subset -> {kept_out} ({len(kept)}/{len(df)} rows kept)")
    return kept_out


# ---------- 5. Orchestration ----------

def curate(path: Path, text_column: str, model: str) -> None:
    print(f"Loading {path} ...")
    df = load_dataset(path, text_column)
    print(f"  {len(df)} rows loaded.")

    print(f"Generating embeddings via Ollama ({model}) ...")
    embeddings = embed_texts(df[text_column].tolist(), model=model)
    print(f"  embeddings shape: {embeddings.shape}")   # e.g. (N, 1024)

    print("Reducing to 2D with UMAP ...")
    coords = project_2d(embeddings)
    df["x"] = coords[:, 0]
    df["y"] = coords[:, 1]
    # Spotlight can render the full vector as a heatmap when typed as Embedding.
    df["embedding"] = list(embeddings)

    # Pick a column to color the scatter by — prefer a categorical label if
    # the user has one (e.g. "source", "label", "category"), else color by
    # the curation flags so toggled rows stand out.
    color_candidates = ["source", "label", "category", "class", "flag", "keep"]
    color_col = next((c for c in color_candidates if c in df.columns), None)

    # Build an explicit 3-pane layout so the scatter plot is the headline view,
    # not buried behind the table.
    custom_layout = sl.layout(
        sl.split(
            sl.scatterplot(
                name="UMAP projection",
                x_column="x",
                y_column="y",
                color_by_column=color_col,
            ),
            sl.split(
                sl.inspector(name="Row inspector"),
                sl.table(name="All rows"),
                orientation="vertical",
                weight=1,
            ),
            orientation="horizontal",
            weight=2,   # give the scatter ~2/3 of the width
        ),
    )

    # Write the editable curated file BEFORE opening Spotlight, since the
    # Community Edition table is read-only and the user has to do the edits
    # in their own text editor.
    curated_path = write_curated_template(df, path)

    print("Launching Spotlight. The browser table is READ-ONLY (CE limit).")
    print("  Workflow:")
    print(f"   1. Open {curated_path.name} in your editor (VSCode / Excel / Notepad++).")
    print("   2. Use Spotlight's scatter to find rows you want to drop.")
    print("      Lasso-select on the scatter -> inspector shows raw text -> note the row id.")
    print("   3. In your editor, change 'keep' from true to false on those rows. Save.")
    print("   4. Close the Spotlight browser tab when done.")
    spotlight.show(
        df,
        dtype={"embedding": spotlight.Embedding},
        layout=custom_layout,
        wait=True,   # blocks until the user closes Spotlight
    )

    print("Spotlight closed. Reading your edits back ...")
    apply_edits_and_save_kept(path)
    print("Done. Your dataset has been molded.")


# ---------- CLI ----------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Visual data curation with Spotlight + Ollama.")
    p.add_argument("--input", "-i", required=True, type=Path,
                   help="Path to a .csv or .jsonl file.")
    p.add_argument("--text-column", "-c", default="text",
                   help="Name of the column holding the text to embed (default: text).")
    p.add_argument("--model", "-m", default=DEFAULT_EMBED_MODEL,
                   help=f"Ollama embedding model name (default: {DEFAULT_EMBED_MODEL}).")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if not args.input.exists():
        print(f"Input file not found: {args.input}", file=sys.stderr)
        return 1
    try:
        curate(args.input, args.text_column, args.model)
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        return 130
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
