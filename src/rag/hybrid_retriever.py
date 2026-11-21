import os
import glob
import re
import numpy as np
from dataclasses import dataclass
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer


@dataclass
class RetrievedChunk:
    doc_id: str
    source_file: str
    content: str
    rrf_score: float
    rerank_score: float


class HybridRetriever:
    """
    Memory-efficient Hybrid RAG Retriever:
    1. Lexical search via BM25 (exact error codes, SOP IDs)
    2. Semantic search via shared Sentence-Transformer embeddings
    3. Reciprocal Rank Fusion (RRF) + exact token-overlap boost reranking
    """

    def __init__(self, docs_dir: str = "data/docs", embedder: SentenceTransformer | None = None):
        self.docs_dir = docs_dir
        self.embedder = embedder or SentenceTransformer("all-MiniLM-L6-v2")
        self.chunks: list[dict] = []
        self._load_and_index()

    def _tokenize(self, text: str) -> list[str]:
        return re.findall(r"\w+", text.lower())

    def _load_and_index(self):
        md_files = sorted(glob.glob(os.path.join(self.docs_dir, "*.md")))
        for filepath in md_files:
            filename = os.path.basename(filepath)
            with open(filepath, "r", encoding="utf-8") as f:
                raw_text = f.read().strip()

            lines = raw_text.splitlines()
            title = lines[0].lstrip("# ").strip() if lines else filename
            for idx, line in enumerate(lines[1:], start=1):
                clean = line.strip()
                if clean.startswith("- "):
                    chunk_text = f"[{title}] {clean[2:]}"
                    self.chunks.append({
                        "doc_id": f"{filename}#L{idx}",
                        "source_file": filename,
                        "content": chunk_text,
                    })

        tokenized_corpus = [self._tokenize(c["content"]) for c in self.chunks]
        self.bm25 = BM25Okapi(tokenized_corpus)

        texts = [c["content"] for c in self.chunks]
        self.dense_matrix = self.embedder.encode(texts, normalize_embeddings=True)

    def retrieve(self, query: str, top_k: int = 3, candidate_pool: int = 8) -> list[RetrievedChunk]:
        if not self.chunks:
            return []

        q_tokens = set(self._tokenize(query))
        bm25_scores = self.bm25.get_scores(list(q_tokens))
        bm25_order = np.argsort(bm25_scores)[::-1]
        bm25_ranks = {int(idx): rank + 1 for rank, idx in enumerate(bm25_order)}

        q_emb = self.embedder.encode([query], normalize_embeddings=True)[0]
        dense_scores = np.dot(self.dense_matrix, q_emb)
        dense_order = np.argsort(dense_scores)[::-1]
        dense_ranks = {int(idx): rank + 1 for rank, idx in enumerate(dense_order)}

        k_rrf = 60
        rrf_scores = {}
        for idx in range(len(self.chunks)):
            rrf_scores[idx] = (1.0 / (k_rrf + bm25_ranks[idx])) + (1.0 / (k_rrf + dense_ranks[idx]))

        top_candidates = sorted(rrf_scores.keys(), key=lambda i: rrf_scores[i], reverse=True)[:candidate_pool]

        # Hybrid rerank score combining dense similarity + exact keyword overlap
        scored = []
        for idx in top_candidates:
            c_tokens = set(self._tokenize(self.chunks[idx]["content"]))
            overlap = len(q_tokens & c_tokens) / max(len(q_tokens), 1)
            rerank_score = float(0.75 * dense_scores[idx] + 0.25 * overlap)
            scored.append((idx, rerank_score))

        reranked = sorted(scored, key=lambda x: x[1], reverse=True)[:top_k]

        return [
            RetrievedChunk(
                doc_id=self.chunks[idx]["doc_id"],
                source_file=self.chunks[idx]["source_file"],
                content=self.chunks[idx]["content"],
                rrf_score=round(rrf_scores[idx], 4),
                rerank_score=round(score, 4),
            )
            for idx, score in reranked
        ]
