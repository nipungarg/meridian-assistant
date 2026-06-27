"""Retrieval pipeline: Chroma top-k -> rerank -> confidence-scored top-n.

``RERANKER`` selects the strategy:
  * ``cross_encoder`` (default): local ``BAAI/bge-reranker-base`` cross-encoder, score via sigmoid.
  * ``llm``: a single structured ChatOpenAI call scores each candidate 0-1.
  * ``none``: rank by cosine similarity (1 - distance).
If the cross-encoder can't be loaded (e.g. offline), it falls back to ``llm`` then ``none``.
The final ``top_score`` drives the low-confidence handoff threshold (``MIN_RETRIEVAL_SCORE``).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

from meridian.config import get_settings
from meridian.ingestion.build_index import get_collection


@dataclass
class RetrievedChunk:
    text: str
    metadata: dict
    score: float
    similarity: float  # raw cosine similarity from the vector store


@dataclass
class RetrievalResult:
    chunks: list[RetrievedChunk] = field(default_factory=list)
    used_reranker: str = "none"
    confidence: float = 0.0  # best embedding similarity among candidates (the gating signal)

    @property
    def top_score(self) -> float:
        return self.chunks[0].score if self.chunks else 0.0

    def is_confident(self, threshold: float | None = None) -> bool:
        # Gate on embedding similarity, NOT the cross-encoder score: ms-marco rerank scores are
        # great for ordering but poorly calibrated for an absolute relevance threshold.
        thr = get_settings().min_retrieval_score if threshold is None else threshold
        return bool(self.chunks) and self.confidence >= thr


def _sigmoid(x: float) -> float:
    try:
        return 1.0 / (1.0 + math.exp(-x))
    except OverflowError:
        return 0.0 if x < 0 else 1.0


class Retriever:
    _cross_encoder = None
    _cross_encoder_failed = False

    def __init__(self) -> None:
        self.settings = get_settings()
        self._collection = None

    @property
    def collection(self):
        if self._collection is None:
            self._collection = get_collection()
        return self._collection

    # ---- cross-encoder (lazy, cached on the class) --------------------------
    @classmethod
    def _get_cross_encoder(cls, model_name: str):
        if cls._cross_encoder is not None or cls._cross_encoder_failed:
            return cls._cross_encoder
        try:
            from sentence_transformers import CrossEncoder
            cls._cross_encoder = CrossEncoder(model_name)
        except Exception:
            cls._cross_encoder_failed = True
            cls._cross_encoder = None
        return cls._cross_encoder

    def retrieve(self, query: str, top_k: int | None = None,
                 top_n: int | None = None) -> RetrievalResult:
        s = self.settings
        top_k = top_k or s.retrieval_top_k
        top_n = top_n or s.rerank_top_n

        res = self.collection.query(query_texts=[query], n_results=top_k)
        docs = res.get("documents", [[]])[0]
        metas = res.get("metadatas", [[]])[0]
        dists = res.get("distances", [[]])[0]
        candidates: list[RetrievedChunk] = []
        for doc, meta, dist in zip(docs, metas, dists):
            sim = max(0.0, 1.0 - float(dist))  # cosine distance -> similarity
            candidates.append(RetrievedChunk(text=doc, metadata=meta, score=sim, similarity=sim))
        if not candidates:
            return RetrievalResult(chunks=[], used_reranker="none")

        mode = s.reranker
        if mode == "cross_encoder":
            rerank, used = self._rerank_cross_encoder(query, candidates)
        elif mode == "llm":
            rerank, used = self._rerank_llm(query, candidates)
        else:
            rerank, used = [c.similarity for c in candidates], "none"

        # Order by a blend of rerank + embedding similarity so a well-calibrated bi-encoder
        # match is never buried by a noisy cross-encoder score. Confidence gates on similarity.
        for c, rk in zip(candidates, rerank):
            c.score = round(0.5 * float(rk) + 0.5 * c.similarity, 4) if used != "none" else c.similarity
        candidates.sort(key=lambda c: c.score, reverse=True)
        confidence = max((c.similarity for c in candidates), default=0.0)
        return RetrievalResult(chunks=candidates[:top_n], used_reranker=used,
                               confidence=round(confidence, 4))

    def _rerank_cross_encoder(self, query, candidates) -> tuple[list[float], str]:
        model = self._get_cross_encoder(self.settings.reranker_model)
        if model is None:
            return self._rerank_llm(query, candidates)  # graceful fallback
        scores = model.predict([(query, c.text) for c in candidates])
        return [_sigmoid(float(s)) for s in scores], "cross_encoder"

    def _rerank_llm(self, query, candidates) -> tuple[list[float], str]:
        try:
            import json

            from langchain_openai import ChatOpenAI
            llm = ChatOpenAI(model=self.settings.openai_chat_model, temperature=0,
                             api_key=self.settings.openai_api_key)
            listing = "\n".join(f"[{i}] {c.text[:500]}" for i, c in enumerate(candidates))
            prompt = (
                "Score how well each passage answers the question on a 0.0-1.0 scale. "
                "Return ONLY JSON: {\"scores\": [..]} with one score per passage, in order.\n\n"
                f"Question: {query}\n\nPassages:\n{listing}"
            )
            raw = llm.invoke(prompt).content
            start, end = raw.find("{"), raw.rfind("}")
            scores = json.loads(raw[start:end + 1]).get("scores", [])
            if len(scores) == len(candidates):
                return [float(s) for s in scores], "llm"
        except Exception:
            pass
        return [c.similarity for c in candidates], "none"


_RETRIEVER: Retriever | None = None


def get_retriever() -> Retriever:
    global _RETRIEVER
    if _RETRIEVER is None:
        _RETRIEVER = Retriever()
    return _RETRIEVER
