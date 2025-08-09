"""Các hàm đánh giá đơn giản cho hệ thống RAG.

Chỉ số triển khai:
- hit_rate@k: Tỷ lệ câu hỏi mà câu trả lời đúng (major kỳ vọng) xuất hiện trong top-k tài liệu truy hồi.
- mrr@k: Mean Reciprocal Rank với rank là vị trí đầu tiên chứa major kỳ vọng (nếu không có -> 0).
- answer_accuracy: Tỷ lệ câu hỏi mà output cuối của RAG chứa tên ngành kỳ vọng.

Giả định ground-truth: với mỗi query ta gán 1 ngành (major) kỳ vọng.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Any
import time


@dataclass
class RetrievalResult:
    query: str
    expected: str
    hit: bool
    rank: int | None  # 1-based rank nếu tìm thấy


@dataclass
class AnswerResult:
    query: str
    expected: str
    answer: str
    correct: bool
    latency_s: float


def evaluate_retrieval(retriever, dataset: List[Dict[str, str]], k: int = 4) -> Dict[str, Any]:
    rows: List[RetrievalResult] = []
    for item in dataset:
        q = item["query"]
        expected = item["expected"].lower()
        docs = retriever.get_relevant_documents(q)
        rank = None
        for idx, d in enumerate(docs[:k]):
            content = (d.page_content or "").lower()
            if expected in content:
                rank = idx + 1
                break
        hit = rank is not None
        rows.append(RetrievalResult(query=q, expected=item["expected"], hit=hit, rank=rank))

    hit_rate = sum(r.hit for r in rows) / len(rows) if rows else 0.0
    mrr = sum((1.0 / r.rank if r.rank else 0.0) for r in rows) / len(rows) if rows else 0.0
    return {
        "rows": rows,
        "hit_rate@k": hit_rate,
        "mrr@k": mrr,
    }


def evaluate_answers(rag_chain, dataset: List[Dict[str, str]]) -> Dict[str, Any]:
    rows: List[AnswerResult] = []
    for item in dataset:
        q = item["query"]
        expected_lower = item["expected"].lower()
        start = time.time()
        try:
            output = rag_chain(q)
            answer = output.get("result", "")
        except Exception as e:  # pragma: no cover - phòng lỗi runtime
            answer = f"<ERROR: {e}>"
        latency = time.time() - start
        correct = expected_lower in answer.lower()
        rows.append(AnswerResult(query=q, expected=item["expected"], answer=answer, correct=correct, latency_s=latency))

    accuracy = sum(r.correct for r in rows) / len(rows) if rows else 0.0
    avg_latency = sum(r.latency_s for r in rows) / len(rows) if rows else 0.0
    return {
        "rows": rows,
        "answer_accuracy": accuracy,
        "avg_latency_s": avg_latency,
    }


def format_seconds(seconds: float) -> str:
    return f"{seconds*1000:.0f} ms" if seconds < 1 else f"{seconds:.2f} s"


def build_default_eval_set() -> List[Dict[str, str]]:
    """Tập nhỏ phục vụ minh họa (có thể mở rộng)."""
    return [
        {
            "query": "Mình thích lập trình phần mềm, tư duy logic tốt và muốn phát triển sản phẩm AI trong tương lai.",
            "expected": "Công nghệ thông tin",
        },
        {
            "query": "Tôi yêu thích sinh học và mong muốn chăm sóc sức khỏe cộng đồng.",
            "expected": "Y dược",
        },
        {
            "query": "Muốn thiết kế nhà cửa và không gian sống sáng tạo hiện đại.",
            "expected": "Kiến trúc",
        },
        {
            "query": "Thích làm việc với trẻ em và đam mê giảng dạy, truyền đạt kiến thức.",
            "expected": "Sư phạm",
        },
        {
            "query": "Giỏi giao tiếp, ngoại ngữ và muốn làm trong lĩnh vực du lịch khách sạn.",
            "expected": "Du lịch và khách sạn",
        },
    ]


if __name__ == "__main__":  # Chạy nhanh ở CLI (yêu cầu biến môi trường & model sẵn)
    from utils import load_vectorstore
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain.chains import RetrievalQA
    import json

    eval_set = build_default_eval_set()
    db = load_vectorstore()
    retriever = db.as_retriever(search_kwargs={"k": 4})
    llm = ChatGoogleGenerativeAI(model="models/gemini-2.5-pro", temperature=0.2)
    chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, return_source_documents=False)

    ret_metrics = evaluate_retrieval(retriever, eval_set, k=4)
    ans_metrics = evaluate_answers(chain, eval_set)

    out = {
        "retrieval": {
            "hit_rate@4": ret_metrics["hit_rate@k"],
            "mrr@4": ret_metrics["mrr@k"],
        },
        "answer": {
            "accuracy": ans_metrics["answer_accuracy"],
            "avg_latency_s": ans_metrics["avg_latency_s"],
        },
    }
    print(json.dumps(out, ensure_ascii=False, indent=2))
