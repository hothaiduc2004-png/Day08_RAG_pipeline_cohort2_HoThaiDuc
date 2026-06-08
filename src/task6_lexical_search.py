"""
Task 6 — Lexical Search Module (BM25).

Mặc định sử dụng BM25. Nếu dùng phương pháp khác (TF-IDF, Elasticsearch,
Weaviate BM25 built-in), hãy giải thích cơ chế trong buổi demo → +5 bonus.

Cài đặt:
    pip install rank-bm25

BM25 hoạt động thế nào:
    - Term Frequency (TF): từ xuất hiện nhiều trong document → điểm cao
    - Inverse Document Frequency (IDF): từ hiếm → quan trọng hơn
    - Document length normalization: document dài không bị ưu tiên quá mức
    - Formula: score(q,d) = Σ IDF(qi) * (tf(qi,d) * (k1+1)) / (tf(qi,d) + k1*(1-b+b*|d|/avgdl))
    - k1=1.5 (term saturation), b=0.75 (length normalization)
"""

import json
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi

VECTOR_STORE_DIR = Path(__file__).parent.parent / "data" / "vector_store"

_bm25_index = None
_store = None


def _load_store() -> dict:
    """Load the local vector store built by task4."""
    global _store
    if _store is not None:
        return _store

    store_path = VECTOR_STORE_DIR / "store.json"
    if not store_path.exists():
        raise FileNotFoundError(
            f"Vector store not found at {store_path}. Run task4_chunking_indexing.py first."
        )

    with store_path.open("r", encoding="utf-8") as f:
        _store = json.load(f)
    return _store


def _tokenize_vietnamese(text: str) -> list[str]:
    """
    Đơn giản split(), hoặc dùng underthesea/PyVi cho việc tokenize tiếng Việt tiên tiến.
    Ở đây dùng regex để tách từ.
    """
    import re

    text = text.lower()
    tokens = re.findall(r"\b[a-z0-9à-ỿ]+\b", text)
    return tokens


def _build_bm25_index():
    """Xây dựng BM25 index từ stored documents."""
    global _bm25_index
    if _bm25_index is not None:
        return _bm25_index

    store = _load_store()
    documents = store.get("documents", [])

    tokenized_corpus = [_tokenize_vietnamese(doc) for doc in documents]
    _bm25_index = BM25Okapi(tokenized_corpus)
    print(f"✓ BM25 index built with {len(tokenized_corpus)} documents")
    return _bm25_index


def lexical_search(query: str, top_k: int = 10) -> list[dict]:
    """
    Tìm kiếm từ khóa sử dụng BM25.

    Args:
        query: Câu truy vấn
        top_k: Số lượng kết quả tối đa

    Returns:
        List of {
            'content': str,
            'score': float,      # BM25 score
            'metadata': dict
        }
        Sorted by score descending.
    """
    bm25 = _build_bm25_index()
    store = _load_store()

    tokenized_query = _tokenize_vietnamese(query)
    scores = bm25.get_scores(tokenized_query)

    top_indices = np.argsort(scores)[::-1][:top_k]

    results = []
    for idx in top_indices:
        if idx < len(store["documents"]):
            results.append(
                {
                    "content": store["documents"][idx],
                    "score": float(scores[idx]),
                    "metadata": store["metadatas"][idx],
                }
            )

    return results


if __name__ == "__main__":
    query = "ma túy hàng không tiếp viên"
    results = lexical_search(query, top_k=5)
    print(f"BM25 Lexical Search: '{query}'")
    print("=" * 60)
    for i, r in enumerate(results, 1):
        print(f"\n[{i}] Score: {r['score']:.3f}")
        print(f"    Source: {r['metadata'].get('source', 'Unknown')}")
        print(f"    Content: {r['content'][:150].replace(chr(10), ' ')}...")
