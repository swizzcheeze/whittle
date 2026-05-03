"""
Embedding cache: SHA256(model || text) -> vector, stored in SQLite next to the dataset.

Re-loads of the same dataset with the same model are instant — only new/changed rows
hit the embedding backend.
"""
from __future__ import annotations

import hashlib
import sqlite3
from pathlib import Path

import numpy as np


def _key(model: str, text: str) -> str:
    h = hashlib.sha256()
    h.update(model.encode("utf-8"))
    h.update(b"\x00")
    h.update(text.encode("utf-8"))
    return h.hexdigest()


class EmbeddingCache:
    """SQLite-backed key->vector store. Vectors are stored as raw float32 bytes."""

    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(path))
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS embeddings ("
            "  key TEXT PRIMARY KEY,"
            "  dim INTEGER NOT NULL,"
            "  vector BLOB NOT NULL"
            ")"
        )
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    def get(self, model: str, text: str) -> np.ndarray | None:
        row = self._conn.execute(
            "SELECT dim, vector FROM embeddings WHERE key = ?",
            (_key(model, text),),
        ).fetchone()
        if row is None:
            return None
        dim, blob = row
        return np.frombuffer(blob, dtype=np.float32, count=dim).copy()

    def put(self, model: str, text: str, vector: np.ndarray) -> None:
        v = np.asarray(vector, dtype=np.float32)
        self._conn.execute(
            "INSERT OR REPLACE INTO embeddings (key, dim, vector) VALUES (?, ?, ?)",
            (_key(model, text), int(v.shape[0]), v.tobytes()),
        )
        self._conn.commit()

    def get_many(self, model: str, texts: list[str]) -> list[np.ndarray | None]:
        return [self.get(model, t) for t in texts]

    def stats(self) -> dict:
        n, = self._conn.execute("SELECT COUNT(*) FROM embeddings").fetchone()
        return {"rows": n, "path": str(self.path)}
