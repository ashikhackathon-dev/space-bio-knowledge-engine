from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np


@dataclass
class VectorIndex:
    dim: int
    index: Any

    def search(self, vectors: np.ndarray, top_k: int) -> Tuple[np.ndarray, np.ndarray]:
        return self.index.search(vectors.astype("float32"), top_k)


def load_faiss_index(vector_dir: str, dim: int) -> Optional[VectorIndex]:
    try:
        import faiss
        import os

        path = os.path.join(vector_dir, "faiss.index")
        if not os.path.exists(path):
            return None
        idx = faiss.read_index(path)
        return VectorIndex(dim=dim, index=idx)
    except Exception:
        return None


def create_empty_faiss(dim: int) -> VectorIndex:
    import faiss

    index = faiss.IndexFlatIP(dim)
    return VectorIndex(dim=dim, index=index)


def add_to_index(vectors: np.ndarray, ids: List[str], vx: VectorIndex) -> None:
    # Store ids in a sidecar mapping file; for now, rely on callers for mapping
    vx.index.add(vectors.astype("float32"))


def persist_faiss_index(vector_dir: str, vx: VectorIndex) -> None:
    import faiss
    import os

    os.makedirs(vector_dir, exist_ok=True)
    path = os.path.join(vector_dir, "faiss.index")
    faiss.write_index(vx.index, path)


def reciprocal_rank_fusion(rankings: List[List[Tuple[str, float]]], k: float = 60.0) -> List[Tuple[str, float]]:
    scores: Dict[str, float] = {}
    for ranked in rankings:
        for rank, (doc_id, _score) in enumerate(ranked, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)



