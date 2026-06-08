"""
Task 9 — Retrieval Pipeline Hoàn Chỉnh.

Kết hợp semantic search + lexical search + reranking + PageIndex fallback
thành một pipeline thống nhất.

Logic:
    1. Chạy semantic_search + lexical_search song song
    2. Merge kết quả (RRF hoặc weighted fusion)
    3. Rerank
    4. Nếu top result score < threshold → fallback sang PageIndex
    5. Return top_k results
"""

try:
    from .task5_semantic_search import semantic_search
    from .task6_lexical_search import lexical_search
    from .task7_reranking import rerank, rerank_rrf
    from .task8_pageindex_vectorless import pageindex_search
except ImportError:
    from task5_semantic_search import semantic_search
    from task6_lexical_search import lexical_search
    from task7_reranking import rerank, rerank_rrf
    from task8_pageindex_vectorless import pageindex_search


# =============================================================================
# CONFIGURATION
# =============================================================================

SCORE_THRESHOLD = 0.3   # Nếu best score < threshold → fallback PageIndex
DEFAULT_TOP_K = 5
RERANK_METHOD = "cross_encoder"  # "cross_encoder" | "mmr" | "rrf"


def retrieve(
    query: str,
    top_k: int = DEFAULT_TOP_K,
    score_threshold: float = SCORE_THRESHOLD,
    use_reranking: bool = True,
) -> list[dict]:
    """
    Retrieval pipeline hoàn chỉnh với fallback logic.

    Pipeline:
        Query
          ├→ Semantic Search → results_dense
          ├→ Lexical Search  → results_sparse
          │
          ├→ Merge (RRF) → merged_results
          ├→ Rerank → reranked_results
          │
          └→ If best_score < threshold:
                └→ PageIndex Vectorless → fallback_results

    Args:
        query: Câu truy vấn
        top_k: Số lượng kết quả cuối cùng
        score_threshold: Ngưỡng điểm tối thiểu cho hybrid results
        use_reranking: Có áp dụng reranking hay không

    Returns:
        List of {
            'content': str,
            'score': float,
            'metadata': dict,
            'source': str  # 'hybrid' hoặc 'pageindex'
        }
    """
    import numpy as np

    print(f"\n{'='*60}")
    print(f"Hybrid Retrieval Pipeline for: '{query}'")
    print(f"{'='*60}")

    # Step 1: Song song chạy semantic + lexical (Lấy gấp đôi top_k để rerank hiệu quả hơn)
    print("\n1️⃣  Semantic Search...")
    try:
        dense_results = semantic_search(query, top_k=top_k * 2)
        print(f"   ✓ Found {len(dense_results)} dense results")
    except Exception as e:
        print(f"   ⚠ Semantic search error: {e}")
        dense_results = []

    print("2️⃣  Lexical Search (BM25)...")
    try:
        sparse_results = lexical_search(query, top_k=top_k * 2)
        print(f"   ✓ Found {len(sparse_results)} sparse results")
    except Exception as e:
        print(f"   ⚠ Lexical search error: {e}")
        sparse_results = []

    # Step 2: Merge bằng RRF
    print(f"\n3️⃣  Merging results via RRF (Reciprocal Rank Fusion)...")
    if dense_results and sparse_results:
        merged = rerank_rrf([dense_results, sparse_results], top_k=top_k * 2)
        for item in merged:
            item["source"] = "hybrid"
        print(f"   ✓ Merged {len(merged)} unique results")
    elif dense_results:
        merged = dense_results
        for item in merged:
            item["source"] = "hybrid"
        print(f"   ✓ Using semantic results only")
    elif sparse_results:
        merged = sparse_results
        for item in merged:
            item["source"] = "hybrid"
        print(f"   ✓ Using lexical results only")
    else:
        merged = []
        print(f"   ⚠ No results from either search method")

    # Step 3: Rerank
    print(f"\n4️⃣  Reranking with {RERANK_METHOD}...")
    if use_reranking and merged:
        try:
            # Chuẩn bị query embedding cho MMR nếu cần
            query_embedding = np.ones(384) if RERANK_METHOD == "mmr" else None
            final_results = rerank(
                query, merged, top_k=top_k, method=RERANK_METHOD, query_embedding=query_embedding
            )
            print(f"   ✓ Reranked to top {len(final_results)} results")
        except Exception as e:
            print(f"   ⚠ Reranking error: {e}. Using merged results.")
            final_results = merged[:top_k]
    else:
        final_results = merged[:top_k]
        print(f"   ⓘ Skipping reranking")

    # Step 4: Check threshold → fallback sang PageIndex Vectorless
    print(f"\n5️⃣  Checking score threshold ({score_threshold})...")
    best_score = final_results[0]["score"] if final_results else 0.0
    
    if not final_results or best_score < score_threshold:
        print(
            f"   ⚠ Best score ({best_score:.3f}) < threshold ({score_threshold}). "
            f"Fallback → PageIndex"
        )
        fallback = pageindex_search(query, top_k=top_k)
        print(f"   ✓ PageIndex returned {len(fallback)} results")
        
        # Đảm bảo source được ghi nhận đúng là nguồn từ pageindex
        for item in fallback:
            item["source"] = "pageindex"
        return fallback

    print(f"   ✓ Score {best_score:.3f} >= {score_threshold} (accept hybrid results)")
    return final_results[:top_k]


if __name__ == "__main__":
    test_queries = [
        "Hình phạt cho tội tàng trữ trái phép chất ma tuý",
        "Nghệ sĩ nào bị bắt vì sử dụng ma tuý năm 2024",
        "Ma túy xách qua đường hàng không",
    ]

    for q in test_queries:
        print(f"\n{'#'*60}")
        print(f"Query: {q}")
        print(f"{'#'*60}")
        results = retrieve(q, top_k=3, score_threshold=0.2)
        print(f"\n🎯 Final Results:")
        for i, r in enumerate(results, 1):
            print(f"  {i}. [{r['score']:.3f}] [{r['source']}] {r['content'][:70].replace(chr(10), ' ')}...")