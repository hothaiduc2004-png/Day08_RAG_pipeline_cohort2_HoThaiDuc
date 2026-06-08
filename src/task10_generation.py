"""
Task 10 — Generation Có Citation.

Hướng dẫn:
    1. Chọn top_k, top_p phù hợp (giải thích lý do)
    2. Sắp xếp lại chunks sau reranking để tránh "lost in the middle"
    3. Inject context vào prompt
    4. Yêu cầu LLM trả lời có citation
    5. Nếu không đủ evidence → "I cannot verify this information"
"""

import os
from dotenv import load_dotenv

load_dotenv()

try:
    from .task9_retrieval_pipeline import retrieve
except ImportError:
    from task9_retrieval_pipeline import retrieve


# =============================================================================
# CONFIGURATION — Giải thích lựa chọn
# =============================================================================

# top_k: Số chunks đưa vào ngữ cảnh (context) của LLM.
# Chọn 5 vì: Đảm bảo cung cấp đủ lượng bằng chứng (evidence) đa chiều từ nhiều nguồn 
# mà không làm tăng quá mức chiều dài prompt, giảm thiểu rủi ro mô hình bị quá tải thông tin.
TOP_K = 5

# top_p (nucleus sampling): Xác suất tích lũy tối đa để lựa chọn tập hợp token tiếp theo.
# Chọn 0.9 vì: Cho phép câu từ có độ tự nhiên, linh hoạt và đa dạng ngữ pháp (diverse) 
# nhưng vẫn nằm trong tầm kiểm soát chặt chẽ, không bị lan man hay sinh chữ ngẫu nhiên.
TOP_P = 0.9

# temperature: Kiểm soát độ sáng tạo/ngẫu nhiên của mô hình.
# Chọn 0.3 vì: Hệ thống RAG cần tính chính xác tối đa về mặt sự thật (factual content), 
# hạn chế tuyệt đối việc LLM tự suy diễn bừa bãi ("ảo tưởng" - hallucination).
TEMPERATURE = 0.3


# =============================================================================
# SYSTEM PROMPT
# =============================================================================

SYSTEM_PROMPT = """Answer the following question comprehensively in Vietnamese based ONLY on the provided context.

For every statement of fact or claim, immediately insert a citation in brackets linking to the specific source name and year if available (e.g., [Luật Phòng chống ma tuý, 2021] or [VnExpress, 2024]).

If the information is not explicitly stated in the provided context or if the context is insufficient to answer the question fully, state exactly: 'I cannot verify this information' rather than guessing or using external knowledge.

Rules:
- Only use information from the provided context. Do not invent, extrapolate, or use external facts.
- Every factual claim MUST have a citation in the exact format: [Nguồn, Năm].
- If evidence is missing, ambiguous, or insufficient, reply with exactly: 'I cannot verify this information'.
- Structure your answer with clear, readable paragraphs."""


# =============================================================================
# DOCUMENT REORDERING (tránh lost in the middle)
# =============================================================================

def reorder_for_llm(chunks: list[dict]) -> list[dict]:
    """
    Sắp xếp chunks để tránh hiệu ứng "lost in the middle".

    LLM ghi nhớ tốt thông tin ở ĐẦU và CUỐI prompt, dễ quên thông tin ở GIỮA.
    Chiến thuật: Đặt các chunk điểm cao nhất ở đầu và cuối chuỗi, các chunk kém hơn ở giữa.

    Input order (giảm dần theo score): [1, 2, 3, 4, 5]
    Output order:                     [1, 3, 5, 4, 2]

    Args:
        chunks: Danh sách các chunk đã được sắp xếp giảm dần theo điểm số từ retrieval.

    Returns:
        Danh sách đã được hoán đổi vị trí để tối ưu khả năng chú ý của LLM.
    """
    if len(chunks) <= 2:
        return chunks

    # Tách thành 2 nhóm dựa trên chỉ mục (index) chẵn/lẻ (0-based)
    evens = [chunks[i] for i in range(0, len(chunks), 2)]  # vị trí 1, 3, 5... (1-based)
    odds = [chunks[i] for i in range(1, len(chunks), 2)]    # vị trí 2, 4, 6... (1-based)
    
    # Đảo ngược nhóm lẻ để đẩy phần tử có điểm cao thứ nhì (index 1) xuống cuối cùng
    odds.reverse()
    
    # Kết hợp: [Chẵn] + [Lẻ đảo ngược] tạo mô hình xen kẽ hình chữ V (ví dụ: 1, 3, 5, 4, 2)
    return evens + odds


# =============================================================================
# CONTEXT FORMATTING
# =============================================================================

def format_context(chunks: list[dict]) -> str:
    """
    Đóng gói các đoạn văn bản (chunks) thành một chuỗi context thống nhất.
    Bổ sung nhãn siêu dữ liệu (metadata) trực quan để LLM làm căn cứ trích dẫn nguồn.

    Args:
        chunks: List gồm các dict chứa {'content': str, 'metadata': dict, 'score': float}

    Returns:
        Chuỗi văn bản context đã được định dạng.
    """
    context_parts = []
    for i, chunk in enumerate(chunks, 1):
        metadata = chunk.get("metadata", {})
        
        # Lấy tên file gốc hoặc tên nguồn phân tách, mặc định là Source + số thứ tự
        source_name = metadata.get("filename", metadata.get("source", f"Source_{i}"))
        doc_type = metadata.get("type", "unknown")
        
        # Làm sạch tên file (bỏ đuôi .md nếu có) để chuỗi trích dẫn nhìn gọn gàng hơn
        if source_name.endswith(".md"):
            source_name = source_name[:-3]
            
        context_parts.append(
            f"[Document {i} | Source: {source_name} | Type: {doc_type}]\n"
            f"{chunk['content'].strip()}\n"
        )
    return "\n---\n".join(context_parts)


# =============================================================================
# GENERATION
# =============================================================================

def generate_with_citation(query: str, top_k: int = TOP_K) -> dict:
    """
    Quy trình RAG End-to-End hoàn chỉnh sinh văn bản kèm trích dẫn nguồn.

    Quy trình:
        1. Gọi pipeline lấy các văn bản liên quan nhất.
        2. Tái cấu trúc thứ tự văn bản tránh 'lost in the middle'.
        3. Định dạng ngữ cảnh kèm nhãn nguồn bài viết.
        4. Thiết lập Prompt toàn diện (system prompt + context + câu hỏi).
        5. Gọi API LLM xử lý.
        6. Trả ra câu trả lời cuối cùng cùng danh sách nguồn đối chiếu.

    Args:
        query: Câu hỏi cần tra cứu từ người dùng.
        top_k: Số lượng kết quả thu thập từ cơ sở dữ liệu.

    Returns:
        Dict phản hồi chuẩn cấu trúc:
        {
            'answer': str,           # Câu trả lời kèm citation dạng [Nguồn, Năm]
            'sources': list[dict],   # Danh sách các chunks thô đã sử dụng
            'retrieval_source': str  # Nguồn truy xuất dữ liệu ('hybrid' hoặc 'pageindex')
        }
    """
    # Step 1: Retrieve dữ liệu từ pipeline của Task 9
    chunks = retrieve(query, top_k=top_k)
    
    if not chunks:
        return {
            "answer": "I cannot verify this information",
            "sources": [],
            "retrieval_source": "none"
        }
    
    # Step 2: Reorder tài liệu nhằm tối ưu hóa sự tập trung của LLM
    reordered_chunks = reorder_for_llm(chunks)
    
    # Step 3: Format cấu trúc context văn bản nguồn
    context_string = format_context(reordered_chunks)
    
    # Step 4: Xây dựng thông điệp gửi cho LLM
    user_message = f"Context:\n{context_string}\n\n---\n\nQuestion: {query}"
    
    # Step 5: Khởi tạo Client và gọi LLM sinh văn bản (Sử dụng OpenAI SDK v1+)
    from openai import OpenAI
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message}
            ],
            temperature=TEMPERATURE,
            top_p=TOP_P,
        )
        answer = response.choices[0].message.content
    except Exception as e:
        print(f"⚠ LLM Generation error: {e}")
        answer = "I cannot verify this information"
        
    # Step 6: Trả về kết quả hoàn chỉnh cho ứng dụng
    return {
        "answer": answer,
        "sources": chunks,
        "retrieval_source": chunks[0].get("source", "hybrid") if chunks else "none"
    }


if __name__ == "__main__":
    test_queries = [
        "Hình phạt cho tội tàng trữ trái phép chất ma tuý theo pháp luật Việt Nam?",
        "Những nghệ sĩ nào đã bị bắt vì liên quan tới ma tuý?",
        "Quy trình cai nghiện bắt buộc theo Luật Phòng chống ma tuý 2021?",
    ]

    for q in test_queries:
        print(f"\n{'='*70}")
        print(f"Q: {q}")
        print("=" * 70)
        result = generate_with_citation(q)
        print(f"\nA: {result['answer']}")
        print(f"\n[Sources: {len(result['sources'])} chunks | via {result['retrieval_source']}]")