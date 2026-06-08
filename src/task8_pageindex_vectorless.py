"""
Task 8 — PageIndex Vectorless RAG.

Đăng ký tài khoản tại: https://pageindex.ai/
SDK & sample code: https://github.com/VectifyAI/PageIndex

PageIndex cho phép RAG mà không cần vector store — sử dụng
structural understanding của document thay vì embedding.

Cài đặt:
    pip install pageindex

Hướng dẫn:
    1. Đăng ký account tại pageindex.ai
    2. Lấy API key
    3. Upload documents
    4. Query sử dụng PageIndex API
"""

import os
import json
from pathlib import Path
from typing import Optional

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

PAGEINDEX_API_KEY = os.getenv("PAGEINDEX_API_KEY", "")
STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"
PAGEINDEX_CACHE = Path(__file__).parent.parent / "data" / "vector_store" / "pageindex_cache.json"


def _load_pageindex_cache() -> dict:
    """Load cached documents for offline testing."""
    if PAGEINDEX_CACHE.exists():
        with PAGEINDEX_CACHE.open("r", encoding="utf-8") as f:
            return json.load(f)
    return {"documents": []}


def _save_pageindex_cache(cache: dict):
    """Save documents cache."""
    PAGEINDEX_CACHE.parent.mkdir(parents=True, exist_ok=True)
    with PAGEINDEX_CACHE.open("w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)


def upload_documents():
    """
    Upload toàn bộ markdown documents lên PageIndex hoặc cache locally.
    """
    cache = {"documents": []}

    # First, cache locally
    for md_file in STANDARDIZED_DIR.rglob("*.md"):
        content = md_file.read_text(encoding="utf-8")
        cache["documents"].append({
            "filename": md_file.name,
            "content": content,
            "type": "legal" if "legal" in md_file.parts else "news",
            "path": str(md_file.relative_to(STANDARDIZED_DIR)),
        })
        print(f"  ✓ Cached: {md_file.name}")

    _save_pageindex_cache(cache)

    # Try to upload to PageIndex if API key is available
    if PAGEINDEX_API_KEY:
        try:
            from pageindex import PageIndex

            pi = PageIndex(api_key=PAGEINDEX_API_KEY)
            for doc in cache["documents"]:
                pi.upload(
                    content=doc["content"],
                    metadata={
                        "filename": doc["filename"],
                        "type": doc["type"],
                    }
                )
                print(f"  ✓ Uploaded to PageIndex: {doc['filename']}")
        except ImportError:
            print("  ⚠ pageindex package not installed. Using local cache.")
        except Exception as e:
            print(f"  ⚠ Failed to upload to PageIndex: {e}. Using local cache.")


def pageindex_search(query: str, top_k: int = 5) -> list[dict]:
    """
    Vectorless retrieval using PageIndex.
    Fallback khi hybrid search không trả về kết quả phù hợp.
    """
    if PAGEINDEX_API_KEY:
        try:
            from pageindex import PageIndex

            pi = PageIndex(api_key=PAGEINDEX_API_KEY)
            results = pi.query(query=query, top_k=top_k)

            return [
                {
                    "content": r.text,
                    "score": r.score,
                    "metadata": r.metadata,
                    "source": "pageindex",
                }
                for r in results
            ]
        except Exception as e:
            print(f"⚠ PageIndex API error: {e}. Falling back to local search.")

    # Fallback: simple keyword matching in local cache
    cache = _load_pageindex_cache()
    documents = cache.get("documents", [])

    if not documents:
        return []

    # Simple TF-based scoring (word overlap)
    query_words = set(query.lower().split())
    scores = []

    for doc in documents:
        content = doc["content"].lower()
        doc_words = set(content.split())
        overlap = len(query_words & doc_words)
        score = overlap / len(query_words) if query_words else 0.0
        if score > 0:
            scores.append({
                "content": doc["content"],
                "score": score,
                "metadata": {"filename": doc["filename"], "type": doc["type"]},
                "source": "pageindex_local_fallback",
            })
            
    # Sắp xếp để đảm bảo lấy top_k điểm cao nhất
    scores.sort(key=lambda x: x["score"], reverse=True)
    return scores[:top_k]


if __name__ == "__main__":
    import sys

    print("=" * 60)
    print("Task 8: PageIndex Vectorless RAG")
    print("=" * 60)

    if not PAGEINDEX_API_KEY:
        print("\n⚠ PAGEINDEX_API_KEY not set.")
        print("  To enable PageIndex integration:")
        print("  1. Register at https://pageindex.ai/")
        print("  2. Get your API key")
        print("  3. Set PAGEINDEX_API_KEY in .env or environment")
        print("\n  Using local cache fallback for now.\n")

    print("Uploading/caching documents...")
    upload_documents()

    print("\nTest query:")
    results = pageindex_search("hình phạt sử dụng ma tuý", top_k=3)
    if results:
        for i, r in enumerate(results, 1):
            print(f"[{i}] Score: {r['score']:.3f} (Source: {r['source']})")
            print(f"    Content: {r['content'][:100].replace(chr(10), ' ')}...")
    else:
        print("  No results found.")