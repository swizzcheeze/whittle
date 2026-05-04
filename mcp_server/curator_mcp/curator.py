"""
Curator: the in-memory state held by the MCP server for the active session.

Holds:
  - the loaded dataset (as a pandas DataFrame, with curation columns added)
  - the embedding matrix (N x D)
  - 2D UMAP coords (N x 2)
  - the dataset's source path (for save_kept output naming)
  - the SQLite embedding cache

Tool functions live in server.py — this module just owns state and the load logic.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from .cache import EmbeddingCache
from .embeddings import EmbeddingClient, EmbeddingConfig, NaNEmbeddingError


CURATION_COLUMNS = ("keep", "flag", "notes")


@dataclass
class LoadedDataset:
    """A snapshot of the active dataset and its derived artifacts."""
    source_path: Path
    text_column: str
    id_column: str | None
    embedding_model: str
    df: pd.DataFrame                        # includes curation columns + (optional) x, y
    embeddings: np.ndarray                  # shape (N, D)
    coords_2d: np.ndarray | None = None     # shape (N, 2) if reduce_to_2d=True


class Curator:
    """Stateful holder for one active dataset. Methods are called by MCP tools."""

    def __init__(self) -> None:
        self.loaded: LoadedDataset | None = None
        self._cache: EmbeddingCache | None = None
        self._client: EmbeddingClient | None = None
        self._viewer = None     # renumics.spotlight.Viewer | None — typed lazily to avoid hard dep

    # ---------- public state ----------

    def is_loaded(self) -> bool:
        return self.loaded is not None

    def require_loaded(self) -> LoadedDataset:
        if self.loaded is None:
            raise RuntimeError("No dataset loaded. Call `load` first.")
        return self.loaded

    # ---------- load pipeline ----------

    def load(
        self,
        path: str | Path,
        text_column: str = "text",
        id_column: str | None = None,
        embedding_config: EmbeddingConfig | None = None,
        reduce_to_2d: bool = False,
        on_progress: Callable[[int, int], None] | None = None,
    ) -> dict:
        """
        Read a CSV or JSONL, embed each row (using the cache where possible),
        optionally project to 2D, and store everything in self.loaded.
        Returns a small summary dict suitable for tool response.
        """
        src = Path(path).expanduser().resolve()
        if not src.exists():
            raise FileNotFoundError(f"Dataset not found: {src}")

        df, text_column = _read_dataset(src, text_column)
        df = _ensure_curation_columns(df)

        config = embedding_config or EmbeddingConfig()
        self._reset_clients(src, config)

        embeddings, cache_hits = self._embed_with_cache(
            df[text_column].tolist(), config.model, on_progress=on_progress
        )

        coords_2d = None
        if reduce_to_2d and len(df) >= 3:
            coords_2d = _project_2d(embeddings)
            df["x"] = coords_2d[:, 0]
            df["y"] = coords_2d[:, 1]

        self.loaded = LoadedDataset(
            source_path=src,
            text_column=text_column,
            id_column=id_column,
            embedding_model=config.model,
            df=df,
            embeddings=embeddings,
            coords_2d=coords_2d,
        )

        return {
            "path": str(src),
            "rows": int(len(df)),
            "text_column": text_column,
            "id_column": id_column,
            "columns": list(df.columns),
            "embedding_model": config.model,
            "embedding_dim": int(embeddings.shape[1]),
            "embeddings_cached_hits": cache_hits,
            "embeddings_freshly_computed": int(len(df) - cache_hits),
            "reduced_to_2d": coords_2d is not None,
        }

    # ---------- read-only inspection tools (Phase 2) ----------

    def get_row(self, row_index: int) -> dict:
        """Return one row as a JSON-serializable dict, keyed by column name."""
        ld = self.require_loaded()
        if row_index < 0 or row_index >= len(ld.df):
            raise IndexError(f"row_index {row_index} out of range [0, {len(ld.df)})")
        # Drop the embedding/x/y noise from the response — caller wants the data, not internals.
        row = ld.df.iloc[row_index].drop(labels=[c for c in ("embedding", "x", "y") if c in ld.df.columns])
        out: dict = {"row_index": int(row_index)}
        for col, val in row.items():
            out[col] = _jsonable(val)
        return out

    def search(self, query: str, k: int = 5) -> list[dict]:
        """Embed the query, return the top-k rows by cosine similarity."""
        ld = self.require_loaded()
        if self._client is None:
            raise RuntimeError("embedding client not initialized; reload the dataset")
        if k <= 0:
            return []
        q_vec = np.asarray(self._client.embed_one(query), dtype=np.float32)
        sims = _cosine_sim_matrix(q_vec[None, :], ld.embeddings)[0]   # shape (N,)
        top = np.argsort(-sims)[: min(k, len(sims))]
        return [self._row_summary(int(i), score=float(sims[i])) for i in top]

    def find_outliers(self, top_k: int = 10, n_neighbors: int = 5) -> list[dict]:
        """Score each row by mean cosine distance to its n_neighbors nearest peers,
        return the top_k most isolated rows (highest mean distance)."""
        ld = self.require_loaded()
        n = len(ld.df)
        if n < 2:
            return []
        n_neighbors = max(1, min(n_neighbors, n - 1))

        sims = _cosine_sim_matrix(ld.embeddings, ld.embeddings)
        np.fill_diagonal(sims, -np.inf)               # ignore self-similarity
        # Top-N similarities per row, then convert to mean distance (1 - mean similarity).
        partitioned = np.partition(sims, -n_neighbors, axis=1)[:, -n_neighbors:]
        mean_sim = partitioned.mean(axis=1)
        outlier_score = 1.0 - mean_sim                # higher = more isolated
        top = np.argsort(-outlier_score)[: min(top_k, n)]
        return [
            self._row_summary(int(i), score=float(outlier_score[i]), score_label="outlier_score")
            for i in top
        ]

    def find_near_duplicates(self, threshold: float = 0.95, top_k: int = 50) -> list[dict]:
        """Return pairs (i, j) with cosine similarity >= threshold, sorted by similarity desc."""
        ld = self.require_loaded()
        n = len(ld.df)
        if n < 2:
            return []
        sims = _cosine_sim_matrix(ld.embeddings, ld.embeddings)
        # Take the strict upper triangle so we don't double-count pairs or include self-pairs.
        iu, ju = np.triu_indices(n, k=1)
        flat = sims[iu, ju]
        mask = flat >= threshold
        pair_i, pair_j, pair_s = iu[mask], ju[mask], flat[mask]
        order = np.argsort(-pair_s)[: min(top_k, len(pair_s))]
        out = []
        for k_idx in order:
            i, j, s = int(pair_i[k_idx]), int(pair_j[k_idx]), float(pair_s[k_idx])
            out.append({
                "row_a": self._row_summary(i),
                "row_b": self._row_summary(j),
                "similarity": s,
            })
        return out

    def _row_summary(
        self,
        idx: int,
        score: float | None = None,
        score_label: str = "score",
        snippet_chars: int = 240,
    ) -> dict:
        """Compact, LLM-friendly row preview."""
        ld = self.require_loaded()
        row = ld.df.iloc[idx]
        text = str(row[ld.text_column])
        snippet = text if len(text) <= snippet_chars else text[: snippet_chars - 1] + "…"
        out: dict = {
            "row_index": int(idx),
            ld.text_column: snippet,
            "keep": bool(row["keep"]),
            "flag": bool(row["flag"]),
        }
        if ld.id_column and ld.id_column in row.index:
            out[ld.id_column] = _jsonable(row[ld.id_column])
        if score is not None:
            out[score_label] = score
        return out

    # ---------- 2D projection + viewer (Phase 4) ----------

    def project_2d(self, seed: int = 42) -> dict:
        """Run UMAP on the loaded embeddings, add x/y columns.

        First call in a fresh process pays a ~25s numba JIT cost — that's why this
        isn't part of `load`. Subsequent calls are fast (~0.5s on small datasets).
        """
        ld = self.require_loaded()
        coords = _project_2d(ld.embeddings, seed=seed)
        ld.df["x"] = coords[:, 0]
        ld.df["y"] = coords[:, 1]
        ld.coords_2d = coords
        return {
            "rows": int(len(coords)),
            "x_range": [float(coords[:, 0].min()), float(coords[:, 0].max())],
            "y_range": [float(coords[:, 1].min()), float(coords[:, 1].max())],
        }

    def launch_viewer(self) -> dict:
        """Open Spotlight in a browser as a non-blocking viewer. Auto-runs project_2d if needed."""
        try:
            from renumics import spotlight                      # type: ignore[import-not-found]
            from renumics.spotlight import layout as sl         # type: ignore[import-not-found]
        except ImportError as exc:
            raise RuntimeError(
                "renumics-spotlight is not installed. Run "
                "`pip install -e .[viewer]` in mcp_server/ to enable launch_viewer."
            ) from exc

        ld = self.require_loaded()
        if ld.coords_2d is None:
            self.project_2d()

        # Close any prior viewer so we don't leak ports.
        if self._viewer is not None:
            try:
                self._viewer.close()
            except Exception:                                   # noqa: BLE001 — best-effort cleanup
                pass
            self._viewer = None

        df_for_view = ld.df.copy()
        df_for_view["embedding"] = list(ld.embeddings)

        color_candidates = ["source", "label", "category", "class", "flag", "keep"]
        color_col = next((c for c in color_candidates if c in df_for_view.columns), None)

        custom_layout = sl.layout(
            sl.split(
                sl.scatterplot(
                    name="UMAP projection",
                    x_column="x", y_column="y",
                    color_by_column=color_col,
                ),
                sl.split(
                    sl.inspector(name="Row inspector"),
                    sl.table(name="All rows"),
                    orientation="vertical",
                ),
                orientation="horizontal",
                weight=2,
            ),
        )

        # Build explicit dtype map: Embedding for the vector, str for all object columns
        # so Spotlight's inspector renders them as text instead of guessing a binary type.
        str_cols = [c for c in df_for_view.columns if df_for_view[c].dtype == object and c != "embedding"]
        dtype_map: dict = {c: str for c in str_cols}
        dtype_map["embedding"] = spotlight.Embedding

        viewer = spotlight.show(
            df_for_view,
            dtype=dtype_map,
            layout=custom_layout,
            wait=False,                                         # non-blocking — return URL to caller
            no_browser=False,
        )
        self._viewer = viewer
        return {"url": viewer.url, "host": viewer.host, "port": int(viewer.port)}

    def close_viewer(self) -> dict:
        """Stop the Spotlight server if one is running."""
        if self._viewer is None:
            return {"closed": False, "reason": "no viewer running"}
        try:
            self._viewer.close()
        finally:
            self._viewer = None
        return {"closed": True}

    # ---------- mutation tools (Phase 3) ----------

    def mark(
        self,
        row_indices: list[int],
        keep: bool | None = None,
        flag: bool | None = None,
        notes: str | None = None,
    ) -> dict:
        """Apply curation flags to one or more rows. Fields left as None are not touched."""
        ld = self.require_loaded()
        n = len(ld.df)
        bad = [i for i in row_indices if i < 0 or i >= n]
        if bad:
            raise IndexError(f"row indices out of range [0, {n}): {bad}")
        if keep is None and flag is None and notes is None:
            raise ValueError("mark() requires at least one of keep, flag, or notes")

        idx = pd.Index(row_indices)
        if keep is not None:
            ld.df.loc[idx, "keep"] = bool(keep)
        if flag is not None:
            ld.df.loc[idx, "flag"] = bool(flag)
        if notes is not None:
            ld.df.loc[idx, "notes"] = str(notes)

        return {
            "updated_rows": len(row_indices),
            "kept_count": int(ld.df["keep"].sum()),
            "flagged_count": int(ld.df["flag"].sum()),
            "noted_count": int((ld.df["notes"].notna() & (ld.df["notes"].astype(str).str.strip() != "")).sum()),
        }

    def save_kept(self, output_path: str | Path | None = None) -> dict:
        """Write rows where keep == True to disk. Default path: <source>.kept.<ext>."""
        ld = self.require_loaded()
        out = Path(output_path).expanduser().resolve() if output_path else \
              ld.source_path.with_suffix(".kept" + ld.source_path.suffix.lower())

        kept = ld.df[ld.df["keep"]].drop(
            columns=[c for c in ("embedding", "x", "y", "keep") if c in ld.df.columns]
        )
        _write_dataframe(kept, out)
        return {"path": str(out), "rows": int(len(kept)), "total": int(len(ld.df))}

    def save_curated(self, output_path: str | Path | None = None) -> dict:
        """Write the full annotated dataset (keeps `keep`/`flag`/`notes` for review)."""
        ld = self.require_loaded()
        out = Path(output_path).expanduser().resolve() if output_path else \
              ld.source_path.with_suffix(".curated" + ld.source_path.suffix.lower())

        annotated = ld.df.drop(columns=[c for c in ("embedding",) if c in ld.df.columns])
        _write_dataframe(annotated, out)
        return {"path": str(out), "rows": int(len(annotated))}

    # ---------- status ----------

    def status(self) -> dict:
        """Snapshot of what's currently loaded."""
        if self.loaded is None:
            return {"loaded": False}
        ld = self.loaded
        df = ld.df
        return {
            "loaded": True,
            "path": str(ld.source_path),
            "rows": int(len(df)),
            "text_column": ld.text_column,
            "id_column": ld.id_column,
            "columns": list(df.columns),
            "embedding_model": ld.embedding_model,
            "embedding_dim": int(ld.embeddings.shape[1]),
            "reduced_to_2d": ld.coords_2d is not None,
            "kept_count": int(df["keep"].sum()),
            "flagged_count": int(df["flag"].sum()),
            "noted_count": int((df["notes"].astype(str).str.len() > 0).sum()),
            "cache": self._cache.stats() if self._cache else None,
        }

    def close(self) -> None:
        if self._viewer is not None:
            try:
                self._viewer.close()
            except Exception:                                   # noqa: BLE001
                pass
            self._viewer = None
        if self._cache is not None:
            self._cache.close()
            self._cache = None
        if self._client is not None:
            self._client.close()
            self._client = None

    # ---------- internals ----------

    def _reset_clients(self, src: Path, config: EmbeddingConfig) -> None:
        if self._cache is not None:
            self._cache.close()
        if self._client is not None:
            self._client.close()
        cache_path = src.with_suffix(src.suffix + ".embcache.sqlite")
        self._cache = EmbeddingCache(cache_path)
        self._client = EmbeddingClient(config)

    def _embed_with_cache(
        self,
        texts: list[str],
        model: str,
        on_progress: Callable[[int, int], None] | None = None,
    ) -> tuple[np.ndarray, int]:
        assert self._cache is not None and self._client is not None
        if not texts:
            return np.empty((0, 0), dtype=np.float32), 0
        cached = self._cache.get_many(model, texts)
        hits = sum(1 for v in cached if v is not None)

        missing_idx = [i for i, v in enumerate(cached) if v is None]
        done = hits  # cache hits already count as done
        total = len(texts)
        if on_progress and done:
            on_progress(done, total)

        if missing_idx:
            if self._client.config.backend == "openai":
                # Batch for OpenAI: /v1/embeddings accepts arrays.
                # Chunk by batch_size so each chunk is cached on completion —
                # partial progress survives an interruption mid-dataset.
                batch_size = self._client.config.batch_size
                for start in range(0, len(missing_idx), batch_size):
                    chunk_idx = missing_idx[start : start + batch_size]
                    chunk_vecs = self._client.embed_batch([texts[i] for i in chunk_idx])
                    for i, raw in zip(chunk_idx, chunk_vecs):
                        v = np.asarray(raw, dtype=np.float32)
                        self._cache.put(model, texts[i], v)
                        cached[i] = v
                    done += len(chunk_idx)
                    if on_progress:
                        on_progress(done, total)
            else:
                # Ollama: one at a time — caches each vector immediately so
                # partial progress survives VRAM pressure or server restarts.
                _dim: int | None = None
                for i in missing_idx:
                    try:
                        raw = self._client.embed_one(texts[i])
                        v = np.asarray(raw, dtype=np.float32)
                        _dim = len(raw)
                    except NaNEmbeddingError:
                        # Model produced NaN for this text — use zero vector so
                        # the run continues. The row will appear as a far outlier.
                        if _dim is None:
                            known = next((c for c in cached if c is not None), None)
                            _dim = len(known) if known is not None else 1024
                        v = np.zeros(_dim, dtype=np.float32)
                        print(
                            f"[whittle] WARNING: NaN embedding at index {i} "
                            f"({texts[i][:60]!r}) — substituted zero vector",
                            file=sys.stderr, flush=True,
                        )
                    self._cache.put(model, texts[i], v)
                    cached[i] = v
                    done += 1
                    if on_progress:
                        on_progress(done, total)

        # Stack — by now every slot is non-None.
        return np.vstack(cached).astype(np.float32), hits


# ---------- helpers ----------

_TEXT_COLUMN_FALLBACKS = (
    "text", "content", "body", "title", "headline",
    "description", "summary", "message", "sentence", "query",
)


def _read_dataset(path: Path, text_column: str) -> tuple[pd.DataFrame, str]:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        df = pd.read_csv(path)
    elif suffix in {".jsonl", ".ndjson"}:
        df = pd.read_json(path, lines=True)
    elif suffix == ".json":
        df = pd.read_json(path)
    elif suffix == ".parquet":
        df = pd.read_parquet(path)
    elif suffix in {".txt", ".md"}:
        return _read_plaintext(path), "text"
    elif suffix == ".pdf":
        return _read_pdf(path), "text"
    elif suffix == ".docx":
        return _read_docx(path), "text"
    else:
        raise ValueError(
            f"Unsupported file type: {suffix!r}. "
            "Supported: .csv  .json  .jsonl  .parquet  .txt  .md  .pdf  .docx"
        )

    # Auto-detect when the requested column is absent.
    if text_column not in df.columns:
        detected = next((c for c in _TEXT_COLUMN_FALLBACKS if c in df.columns), None)
        if detected is None:
            str_cols = [c for c in df.columns if df[c].dtype == object]
            detected = max(str_cols, key=lambda c: df[c].astype(str).str.len().mean()) if str_cols else None
        if detected is None:
            raise KeyError(
                f"Column '{text_column}' not found. Available: {list(df.columns)}"
            )
        text_column = detected

    df = df.dropna(subset=[text_column]).reset_index(drop=True)
    df[text_column] = df[text_column].astype(str)
    return df, text_column


def _read_plaintext(path: Path) -> pd.DataFrame:
    text = path.read_text(encoding="utf-8", errors="replace")
    # Try paragraph-splitting first; fall back to line-splitting for short files.
    paras = [p.strip() for p in text.split("\n\n") if p.strip()]
    if len(paras) < 3:
        paras = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not paras:
        raise ValueError(f"No text content found in {path.name}.")
    return pd.DataFrame({"text": paras})


def _read_pdf(path: Path) -> pd.DataFrame:
    try:
        from pypdf import PdfReader          # type: ignore[import-not-found]
    except ImportError as exc:
        raise RuntimeError(
            "pypdf is required for PDF files. "
            "Run `pip install pypdf` inside mcp_server/ to enable it."
        ) from exc
    reader = PdfReader(path)
    rows = []
    for i, page in enumerate(reader.pages):
        text = (page.extract_text() or "").strip()
        if text:
            rows.append({"page": i + 1, "text": text})
    if not rows:
        raise ValueError(
            f"No extractable text in {path.name} — "
            "the PDF may be scanned/image-only (OCR not supported)."
        )
    return pd.DataFrame(rows)


def _read_docx(path: Path) -> pd.DataFrame:
    try:
        from docx import Document            # type: ignore[import-not-found]
    except ImportError as exc:
        raise RuntimeError(
            "python-docx is required for .docx files. "
            "Run `pip install python-docx` inside mcp_server/ to enable it."
        ) from exc
    doc = Document(path)
    paras = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
    if not paras:
        raise ValueError(f"No text content found in {path.name}.")
    return pd.DataFrame({"text": paras})


def _ensure_curation_columns(df: pd.DataFrame) -> pd.DataFrame:
    if "keep" not in df.columns:
        df["keep"] = True
    if "flag" not in df.columns:
        df["flag"] = False
    if "notes" not in df.columns:
        df["notes"] = ""
    return df


def _write_dataframe(df: pd.DataFrame, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    suffix = out.suffix.lower()
    if suffix == ".csv":
        df.to_csv(out, index=False)
    elif suffix in {".jsonl", ".ndjson"}:
        df.to_json(out, orient="records", lines=True, force_ascii=False)
    elif suffix == ".json":
        df.to_json(out, orient="records", force_ascii=False)
    else:
        raise ValueError(f"Unsupported output format: {suffix}")


def _project_2d(embeddings: np.ndarray, seed: int = 42) -> np.ndarray:
    import umap
    n_neighbors = min(15, max(2, len(embeddings) - 1))
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=0.1,
        metric="cosine",
        random_state=seed,
    )
    return reducer.fit_transform(embeddings)


def _cosine_sim_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Cosine similarity between every row of `a` and every row of `b`. Robust to zero-norm."""
    eps = 1e-12
    a_n = a / (np.linalg.norm(a, axis=1, keepdims=True) + eps)
    b_n = b / (np.linalg.norm(b, axis=1, keepdims=True) + eps)
    return a_n @ b_n.T


def _jsonable(value):
    """Coerce numpy/pandas scalars to plain Python types for JSON serialization."""
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if pd.isna(value):
        return None
    return value
