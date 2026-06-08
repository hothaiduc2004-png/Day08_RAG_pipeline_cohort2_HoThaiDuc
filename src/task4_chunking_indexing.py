"""
Task 4 — Chunking & Indexing vào Vector Store.

Hướng dẫn:
    1. Đọc toàn bộ markdown files từ data/standardized/
    2. Chọn 1 chunking strategy (giải thích lý do)
    3. Chọn 1 embedding model (giải thích lý do)
    4. Index vào vector store (Weaviate khuyến cáo)

Chunking options (langchain-text-splitters):
    - RecursiveCharacterTextSplitter: an toàn, phổ biến
    - MarkdownHeaderTextSplitter: tốt cho file có heading
    - SemanticChunker: dùng embedding để tách (nâng cao)

Embedding model options:
    - sentence-transformers/all-MiniLM-L6-v2 (384 dim, nhẹ)
    - BAAI/bge-m3 (1024 dim, multilingual, tốt cho tiếng Việt)
    - OpenAI text-embedding-3-small (1536 dim, API)

Vector store options:
    - Weaviate (khuyến cáo: hỗ trợ hybrid search built-in)
    - ChromaDB (đơn giản, local)
    - FAISS (chỉ dense search)

Cài đặt:
    pip install langchain-text-splitters sentence-transformers weaviate-client
"""

import json
import os
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"
VECTOR_STORE_DIR = Path(__file__).parent.parent / "data" / "vector_store"
VECTOR_STORE_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# CONFIGURATION — Giải thích lựa chọn của bạn trong comment
# =============================================================================

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
CHUNKING_METHOD = "recursive"  # "recursive" | "markdown_header" | "semantic"

# Chọn sentence-transformers/all-MiniLM-L6-v2 vì nhẹ, nhanh và phù hợp cho tiếng Việt.
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384

# Chọn vector store cục bộ ChromaDB nếu có, còn không sẽ lưu fallback tại data/vector_store/
VECTOR_STORE = "chromadb"  # "weaviate" | "chromadb" | "local"


def load_documents() -> list[dict]:
    """
    Đọc toàn bộ markdown files từ data/standardized/.

    Returns:
        List of {'content': str, 'metadata': {'source': str, 'path': str, 'type': str}}
    """
    documents = []
    for md_file in STANDARDIZED_DIR.rglob("*.md"):
        content = md_file.read_text(encoding="utf-8")
        doc_type = "legal" if "legal" in md_file.parts else "news" if "news" in md_file.parts else "other"
        documents.append({
            "content": content,
            "metadata": {
                "source": md_file.name,
                "path": str(md_file.relative_to(STANDARDIZED_DIR)),
                "type": doc_type,
            },
        })
    return documents


def chunk_documents(documents: list[dict]) -> list[dict]:
    """
    Chunk documents theo strategy đã chọn.

    Returns:
        List of {'content': str, 'metadata': dict} — mỗi item là 1 chunk
    """
    if CHUNKING_METHOD == "markdown_header":
        from langchain_text_splitters import MarkdownHeaderTextSplitter

        splitter = MarkdownHeaderTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
        )
    else:
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

    chunks = []
    for doc_index, doc in enumerate(documents):
        splits = splitter.split_text(doc["content"])
        for chunk_index, chunk_text in enumerate(splits):
            chunks.append({
                "content": chunk_text,
                "metadata": {
                    **doc["metadata"],
                    "chunk_index": chunk_index,
                    "document_index": doc_index,
                },
            })
    return chunks


def embed_chunks(chunks: list[dict]) -> list[dict]:
    """
    Embed toàn bộ chunks bằng model đã chọn.

    Returns:
        Mỗi chunk dict được thêm key 'embedding': list[float]
    """
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(
        EMBEDDING_MODEL,
        device="cpu",
    )
    texts = [chunk["content"] for chunk in chunks]
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)

    for chunk, embedding in zip(chunks, embeddings):
        chunk["embedding"] = embedding.astype(float).tolist()
    return chunks


def index_to_vectorstore(chunks: list[dict]):
    """
    Lưu chunks vào vector store đã chọn.
    """
    ids = [f"chunk_{i:06d}" for i in range(len(chunks))]
    documents = [chunk["content"] for chunk in chunks]
    metadatas = [chunk["metadata"] for chunk in chunks]
    embeddings = [chunk["embedding"] for chunk in chunks]

    try:
        import chromadb
        from chromadb.config import Settings

        client = chromadb.Client(Settings(chroma_db_impl="duckdb+parquet", persist_directory=str(VECTOR_STORE_DIR)))
        collection = client.get_or_create_collection(name="drug_article_chunks")

        collection.add(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
            embeddings=embeddings,
        )
        client.persist()
        print(f"✓ Indexed {len(chunks)} chunks into ChromaDB at {VECTOR_STORE_DIR}")
        return
    except ImportError:
        print("⚠ chromadb không sẵn sàng. Dùng fallback vector store cục bộ.")

    store = {
        "ids": ids,
        "documents": documents,
        "metadatas": metadatas,
    }
    (VECTOR_STORE_DIR / "store.json").write_text(json.dumps(store, ensure_ascii=False, indent=2), encoding="utf-8")

    try:
        import numpy as np

        np.save(VECTOR_STORE_DIR / "embeddings.npy", np.array(embeddings, dtype="float32"))
        print(f"✓ Indexed {len(chunks)} chunks into local fallback store at {VECTOR_STORE_DIR}")
    except ImportError:
        raise RuntimeError("Không thể lưu fallback vector store vì thiếu numpy.")


def run_pipeline():
    """Chạy toàn bộ pipeline: load → chunk → embed → index."""
    print("=" * 50)
    print("Task 4: Chunking & Indexing")
    print(f"  Chunking: {CHUNKING_METHOD} (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")
    print(f"  Embedding: {EMBEDDING_MODEL} (dim={EMBEDDING_DIM})")
    print(f"  Vector Store: {VECTOR_STORE}")
    print("=" * 50)

    docs = load_documents()
    print(f"\n✓ Loaded {len(docs)} documents")

    chunks = chunk_documents(docs)
    print(f"✓ Created {len(chunks)} chunks")

    chunks = embed_chunks(chunks)
    print(f"✓ Embedded {len(chunks)} chunks")

    index_to_vectorstore(chunks)
    print("✓ Indexed to vector store")


if __name__ == "__main__":
    run_pipeline()
