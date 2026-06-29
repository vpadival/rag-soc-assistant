"""
eval/reranker.py
----------------
Cross-encoder reranker for the SOC Assistant RAG pipeline.

Implements the second stage of two-stage retrieval:
  1. Bi-encoder (ChromaDB + nomic-embed-text) retrieves a candidate set fast.
  2. Cross-encoder (this module) re-scores candidates with higher accuracy,
     then returns hits sorted by descending relevance score.

The cross-encoder reads the query and each document *together*, so it can
model fine-grained query-document interactions that a bi-encoder misses.

Usage (standalone):
    from eval.reranker import Reranker
    reranker = Reranker()
    reranked_hits = reranker.rerank(query, hits, top_n=2)

Usage (integrated into rag.py):
    hits = retrieve(query, collection, top_k=10)   # cast a wide net
    hits = reranker.rerank(query, hits, top_n=2)   # rerank, keep best 2
    prompt = build_prompt(query, hits)
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

# Type alias matching rag.py
Hit = dict[str, Any]

# Default cross-encoder model — small, fast, well-suited for semantic similarity.
# Swap for "cross-encoder/ms-marco-MiniLM-L-12-v2" for higher accuracy at
# the cost of ~2× inference time.
DEFAULT_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class Reranker:
    """
    Wraps a sentence-transformers CrossEncoder to rerank retrieval hits.

    Parameters
    ----------
    model_name : str
        HuggingFace model identifier for the cross-encoder.
    device : str | None
        PyTorch device string ("cpu", "cuda", "mps"). None = auto-detect.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        device: str | None = None,
    ) -> None:
        try:
            from sentence_transformers import CrossEncoder  # type: ignore[import]
            kwargs: dict[str, Any] = {"device": device} if device is not None else {}
            self._model = CrossEncoder(model_name, **kwargs)
            self._available = True
            logger.info("Reranker loaded model: %s", model_name)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "CrossEncoder unavailable (%s). Falling back to distance-based ranking.",
                exc,
            )
            self._model = None
            self._available = False

    # ── Public API ─────────────────────────────────────────────────────────────

    def rerank(
        self,
        query: str,
        hits: list[Hit],
        top_n: int | None = None,
    ) -> list[Hit]:
        """
        Rerank *hits* for *query* and return them sorted by descending score.

        Parameters
        ----------
        query   : The original alert / user query string.
        hits    : List of Hit dicts from rag.retrieve() — each must have a
                  "document" key containing the passage text.
        top_n   : If given, return only the top_n hits after reranking.
                  If None, return all hits in reranked order.

        Returns
        -------
        List of Hit dicts, each augmented with a "rerank_score" float key,
        sorted best-first. Original ChromaDB "distance" is preserved.
        """
        if not hits:
            return hits

        if self._available and self._model is not None:
            reranked = self._cross_encoder_rerank(query, hits)
        else:
            reranked = self._distance_fallback(hits)

        return reranked[:top_n] if top_n is not None else reranked

    # ── Internal strategies ────────────────────────────────────────────────────

    def _cross_encoder_rerank(self, query: str, hits: list[Hit]) -> list[Hit]:
        """Score (query, document) pairs with the cross-encoder."""
        pairs = [(query, h["document"]) for h in hits]
        scores: list[float] = self._model.predict(pairs).tolist()  # type: ignore[union-attr]

        scored = [
            {**h, "rerank_score": float(score)}
            for h, score in zip(hits, scores)
        ]
        return sorted(scored, key=lambda x: x["rerank_score"], reverse=True)

    @staticmethod
    def _distance_fallback(hits: list[Hit]) -> list[Hit]:
        """
        Fallback: sort by ChromaDB cosine distance (lower = more similar).
        Adds a synthetic rerank_score = 1 - distance so higher is still better.
        """
        scored = [
            {**h, "rerank_score": 1.0 - h.get("distance", 1.0)}
            for h in hits
        ]
        return sorted(scored, key=lambda x: x["rerank_score"], reverse=True)


# ── Module-level convenience instance ─────────────────────────────────────────

_default_reranker: Reranker | None = None


def get_default_reranker() -> Reranker:
    """Return (and lazily create) a module-level Reranker singleton."""
    global _default_reranker
    if _default_reranker is None:
        _default_reranker = Reranker()
    return _default_reranker


def rerank(
    query: str,
    hits: list[Hit],
    top_n: int | None = None,
) -> list[Hit]:
    """
    Module-level shortcut — reranks using the default singleton Reranker.

    Example
    -------
    from eval.reranker import rerank
    best_hits = rerank(query, hits, top_n=2)
    """
    return get_default_reranker().rerank(query, hits, top_n=top_n)


# ── CLI smoke-test ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    from pathlib import Path

    ROOT = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(ROOT))

    import chromadb
    import rag

    CHROMA_PATH = str(ROOT / "chroma_store")
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = client.get_collection(rag.COLLECTION)

    query = "Multiple failed SSH logins from external IP targeting root"
    print(f"\nQuery: {query}\n")

    # Retrieve a wider candidate set, then rerank down to 2
    hits = rag.retrieve(query, collection, top_k=6)
    print("Before reranking (by distance):")
    for h in hits:
        print(f"  [{h['id']}] dist={h['distance']:.4f}  {h['metadata']['title']}")

    reranker = Reranker()
    reranked = reranker.rerank(query, hits, top_n=2)
    print("\nAfter reranking (top 2):")
    for h in reranked:
        print(
            f"  [{h['id']}] score={h['rerank_score']:.4f}  {h['metadata']['title']}"
        )