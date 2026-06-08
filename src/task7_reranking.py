"""
Task 7 — Reranking Module.

Chọn 1 trong các phương pháp:
    - Cross-encoder reranker: Jina Reranker v2 (multilingual) hoặc Qwen3-Reranker
    - MMR (Maximal Marginal Relevance): tự implement
    - RRF (Reciprocal Rank Fusion): tự implement

Nếu dùng MMR hoặc RRF, đảm bảo hiểu và giải thích được cơ chế.
"""

import numpy as np
from typing import Optional


def _cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(vec_a, vec_b) / (norm_a * norm_b))


def rerank_mmr(
    query_embedding: np.ndarray,
    candidates: list[dict],
    top_k: int = 5,
    lambda_param: float = 0.7,
) -> list[dict]:
    """
    Maximal Marginal Relevance — chọn candidates vừa relevant vừa diverse.

    MMR = λ * sim(query, doc) - (1-λ) * max(sim(doc, selected_docs))

    Args:
        query_embedding: Vector embedding của query (numpy array)
        candidates: List of {'content': str, 'score': float, 'embedding': list, 'metadata': dict}
        top_k: Số lượng kết quả
        lambda_param: Trade-off giữa relevance (1.0) và diversity (0.0)

    Returns:
        List of top_k candidates selected by MMR.
    """
    if not candidates or len(candidates) == 0:
        return []

    selected = []
    remaining = set(range(len(candidates)))

    # Convert embeddings to numpy arrays
    embeddings = []
    for cand in candidates:
        if "embedding" in cand:
            embeddings.append(np.array(cand["embedding"], dtype=np.float32))
        else:
            # Fallback: use original score as pseudo-embedding dimension
            embeddings.append(np.array([cand.get("score", 0.0)], dtype=np.float32))

    # If all candidates only have scalar fallback embeddings, normalize query_embedding to scalar too
    if all(e.ndim == 1 and e.shape[0] == 1 for e in embeddings):
        if query_embedding.ndim != 1 or query_embedding.shape[0] != 1:
            query_embedding = np.ones(1, dtype=np.float32)

    for _ in range(min(top_k, len(candidates))):
        best_idx = None
        best_score = float("-inf")

        for idx in remaining:
            # Relevance to query
            relevance = _cosine_similarity(query_embedding, embeddings[idx])

            # Max similarity to already selected docs
            max_sim_to_selected = 0
            for sel_idx in selected:
                sim = _cosine_similarity(embeddings[idx], embeddings[sel_idx])
                max_sim_to_selected = max(max_sim_to_selected, sim)

            # MMR score
            mmr_score = lambda_param * relevance - (1 - lambda_param) * max_sim_to_selected

            if mmr_score > best_score:
                best_score = mmr_score
                best_idx = idx

        if best_idx is not None:
            selected.append(best_idx)
            remaining.remove(best_idx)

    return [candidates[i] for i in selected]


def rerank_rrf(
    ranked_lists: list[list[dict]], top_k: int = 5, k: int = 60
) -> list[dict]:
    """
    Reciprocal Rank Fusion — gộp kết quả từ nhiều ranker.

    RRF(d) = Σ 1 / (k + rank_r(d))

    Args:
        ranked_lists: List of ranked result lists (mỗi list từ 1 ranker)
        top_k: Số lượng kết quả cuối cùng
        k: Smoothing constant (default=60, từ paper Cormack et al. 2009)

    Returns:
        List of top_k candidates sorted by RRF score descending.
    """
    rrf_scores = {}  # content -> score
    content_map = {}  # content -> full dict

    for ranked_list in ranked_lists:
        for rank, item in enumerate(ranked_list, 1):
            key = item.get("content", str(item))
            rrf_scores[key] = rrf_scores.get(key, 0) + 1 / (k + rank)
            if key not in content_map:
                content_map[key] = item.copy()

    # Sort by RRF score
    sorted_items = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

    results = []
    for content, score in sorted_items[:top_k]:
        item = content_map[content].copy()
        item["score"] = score
        results.append(item)

    return results


# =============================================================================
# Main rerank interface
# =============================================================================

def rerank(
    query: str,
    candidates: list[dict],
    top_k: int = 5,
    method: str = "mmr",  # "mmr" | "rrf"
    query_embedding: Optional[np.ndarray] = None,
    lambda_param: float = 0.7,
) -> list[dict]:
    """
    Unified reranking interface.

    Args:
        query: Câu truy vấn (dùng cho RRF)
        candidates: Danh sách candidates từ retrieval
        top_k: Số lượng kết quả sau rerank
        method: Phương pháp reranking ("mmr" | "rrf")
        query_embedding: Query embedding vector (required for MMR)
        lambda_param: MMR trade-off parameter

    Returns:
        List of top_k reranked candidates.
    """
    if method == "mmr":
        if query_embedding is None:
            # Fallback: use first candidate's embedding as dummy query
            if candidates and "embedding" in candidates[0]:
                query_embedding = np.array(candidates[0]["embedding"], dtype=np.float32)
            else:
                query_embedding = np.ones(384, dtype=np.float32)
        else:
            query_embedding = np.array(query_embedding, dtype=np.float32)
        return rerank_mmr(query_embedding, candidates, top_k, lambda_param)
    elif method == "rrf":
        # RRF expects list of ranked lists
        return rerank_rrf([candidates], top_k)
    elif method == "cross_encoder":
        # Cross-encoder reranking is not implemented in this local version.
        # Fall back to MMR when embeddings exist, otherwise use original scores.
        has_embedding = any("embedding" in cand for cand in candidates)
        if has_embedding:
            query_embedding = np.array(query_embedding, dtype=np.float32) if query_embedding is not None else np.ones(384, dtype=np.float32)
            return rerank_mmr(query_embedding, candidates, top_k, lambda_param)
        return sorted(candidates, key=lambda c: c.get("score", 0.0), reverse=True)[:top_k]
    else:
        raise ValueError(f"Unknown rerank method: {method}")


if __name__ == "__main__":
    # Test with dummy data
    dummy_candidates = [
        {
            "content": "Điều 248: Tội tàng trữ trái phép chất ma tuý",
            "score": 0.8,
            "embedding": [0.1] * 384,
            "metadata": {},
        },
        {
            "content": "Nghệ sĩ X bị bắt vì sử dụng ma tuý",
            "score": 0.7,
            "embedding": [0.2] * 384,
            "metadata": {},
        },
        {
            "content": "Hình phạt tù từ 2-7 năm cho tội tàng trữ",
            "score": 0.6,
            "embedding": [0.15] * 384,
            "metadata": {},
        },
    ]
    query_emb = np.array([0.12] * 384)
    results = rerank("hình phạt tàng trữ ma tuý", dummy_candidates, top_k=2, method="mmr", query_embedding=query_emb)
    print("Reranked results (MMR):")
    for r in results:
        print(f"  [{r['score']:.3f}] {r['content'][:60]}...")
