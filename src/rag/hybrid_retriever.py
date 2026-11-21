import os
import glob
import re
import numpy as np
from dataclasses import dataclass
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder


@dataclass
class RetrievedChunk:
    doc_id: str
    source_file: str
    content: str
    rrf_score: float
    rerank_score: float


class HybridRetriever:
    """
    Production Hybrid RAG Retriever:
    1. Lexical search via BM25 (exact error codes, SOP IDs, headers)
    2. Semantic search via Sentence-Transformers dense embeddings
    3. Reciprocal Rank Fusion (RRF) to merge lexical + semantic rankings
    4. Cross-Encoder reranking for high-precision top-k context selection
    """

    def __init__(self, docs_dir: str = "data/docs"):
        self.docs_dir = docs_dir
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")
        self.reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
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
            # Split bullet sections into granular, self-contained chunks
            for idx, line in enumerate(lines[1:], start=1):
                clean = line.strip()
                if clean.startswith("- "):
                    chunk_text = f"[{title}] {clean[2:]}"
                    self.chunks.append({
                        "doc_id": f"{filename}#L{idx}",
                        "source_file": filename,
                        "content": chunk_text,
                    })

        # Build BM25 lexical index
        tokenized_corpus = [self._tokenize(c["content"]) for c in self.chunks]
        self.bm25 = BM25Okapi(tokenized_corpus)

        # Build Dense Vector index
        texts = [c["content"] for c in self.chunks]
        self.dense_matrix = self.embedder.encode(texts, normalize_embeddings=True)

    def retrieve(self, query: str, top_k: int = 3, candidate_pool: int = 8) -> list[RetrievedChunk]:
        if not self.chunks:
            return []

        # 1. BM25 scores & ranks
        bm25_scores = self.bm25.get_scores(self._tokenize(query))
        bm25_order = np.argsort(bm25_scores)[::-1]
        bm25_ranks = {int(idx): rank + 1 for rank, idx in enumerate(bm25_order)}

        # 2. Dense cosine similarity scores & ranks
        q_emb = self.embedder.encode([query], normalize_embeddings=True)[0]
        dense_scores = np.dot(self.dense_matrix, q_emb)
        dense_order = np.argsort(dense_scores)[::-1]
        dense_ranks = {int(idx): rank + 1 for rank, idx in enumerate(dense_order)}

        # 3. Reciprocal Rank Fusion (RRF, k=60)
        k_rrf = 60
        rrf_scores = {}
        for idx in range(len(self.chunks)):
            rrf_scores[idx] = (1.0 / (k_rrf + bm25_ranks[idx])) + (1.0 / (k_rrf + dense_ranks[idx]))

        top_candidates = sorted(rrf_scores.keys(), key=lambda i: rrf_scores[i], reverse=True)[:candidate_pool]

        # 4. Cross-Encoder Reranking on top candidates
        pairs = [(query, self.chunks[idx]["content"]) for idx in top_candidates]
        ce_scores = self.reranker.predict(pairs)

        reranked = sorted(
            zip(top_candidates, ce_scores),
            key=lambda x: float(x[1]),
            reverse=True,
        )[:top_k]

        results = []
        for idx, ce_score in reranked:
            c = self.chunks[idx]
            results.append(
                RetrievedChunk(
                    doc_id=c["doc_id"],
                    source_file=c["source_file"],
                    content=c["content"],
                    rrf_score=round(rrf_scores[idx], 4),
                    rerank_score=round(float(ce_score), 4),
                )
            )
        return results


if __name__ == "__main__":
    retriever = HybridRetriever()
    q = "Why is our inference endpoint giving HTTP 502 Bad Gateway?"
    hits = retriever.retrieve(q, top_k=3)
    print(f"Query: {q}\n")
    for h in hits:
        print(f"[{h.doc_id}] (rerank={h.rerank_score}) -> {h.content}")