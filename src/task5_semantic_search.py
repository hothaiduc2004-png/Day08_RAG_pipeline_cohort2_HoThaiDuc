"""
Task 5 — Semantic Search Module.

Viết module tìm kiếm ngữ nghĩa (dense retrieval) trên vector store.

Yêu cầu:
    - Input: query string + top_k
    - Output: danh sách chunks có score, sorted descending
    - Phải tương thích với embedding model và vector store ở Task 4
"""


import json
import os
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np
from sentence_transformers import SentenceTransformer

VECTOR_STORE_DIR = Path(__file__).parent.parent / "data" / "vector_store"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
_CHROMA_COLLECTION_NAME = "drug_article_chunks"

_local_store_cache = None
_embedding_model = None


def _load_embedding_model():
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = SentenceTransformer(
            EMBEDDING_MODEL,
            device="cpu",
        )
    return _embedding_model


def _load_local_store() -> dict:
    global _local_store_cache
    if _local_store_cache is not None:
        return _local_store_cache

    store_path = VECTOR_STORE_DIR / "store.json"
    embeddings_path = VECTOR_STORE_DIR / "embeddings.npy"
    if not store_path.exists() or not embeddings_path.exists():
        raise FileNotFoundError(
            "Local vector store chưa được build. Hãy chạy task4_chunking_indexing.py trước."
        )

    with store_path.open("r", encoding="utf-8") as f:
        store = json.load(f)

    store["embeddings"] = np.load(embeddings_path)
    _local_store_cache = store
    return store


def semantic_search(query: str, top_k: int = 10) -> list[dict]:
    """
    Returns:
        List of {'content': str, 'score': float, 'metadata': dict}
    """
    model = _load_embedding_model()
    query_embedding = model.encode(query, convert_to_numpy=True)

    try:
        import chromadb
        from chromadb.config import Settings

        client = chromadb.Client(Settings(chroma_db_impl="duckdb+parquet", persist_directory=str(VECTOR_STORE_DIR)))
        collection = client.get_collection(name=_CHROMA_COLLECTION_NAME)

        results = collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )

        hits = []
        documents = results["documents"][0]
        metadatas = results["metadatas"][0]
        distances = results["distances"][0]
        for doc, metadata, distance in zip(documents, metadatas, distances):
            score = 1.0 - distance
            hits.append({"content": doc, "score": score, "metadata": metadata})
        return hits
    except Exception:
        store = _load_local_store()
        embeddings = store["embeddings"]
        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(1, -1)

        query_norm = np.linalg.norm(query_embedding)
        docs_norm = np.linalg.norm(embeddings, axis=1)
        denom = docs_norm * query_norm
        similarity = np.zeros(len(embeddings), dtype=float)
        mask = denom > 0
        similarity[mask] = np.dot(embeddings[mask], query_embedding) / denom[mask]

        best_idx = np.argsort(similarity)[::-1][:top_k]
        results = []
        for idx in best_idx:
            results.append(
                {
                    "content": store["documents"][idx],
                    "score": float(similarity[idx]),
                    "metadata": store["metadatas"][idx],
                }
            )
        return results


if __name__ == "__main__":
    results = semantic_search("hình phạt cho tội tàng trữ ma túy", top_k=5)
    for r in results:
        snippet = r["content"][:200].replace("\n", " ")
        print(f"[{r['score']:.3f}] {snippet}\n")
