"""End-to-end RAG orchestration."""

from __future__ import annotations

from typing import Any

from ragatouille import RAGPretrainedModel

from tourism_rag_assistant.llm.generator import ReaderLLM, generate_answer


def retrieve_passages(
    question: str,
    knowledge_index: Any,
    reranker: RAGPretrainedModel | None = None,
    num_retrieved_docs: int = 20,
    num_docs_final: int = 5,
) -> list[str]:
    """Retrieve candidate passages and optionally rerank them with ColBERT."""

    docs = knowledge_index.similarity_search(query=question, k=num_retrieved_docs)
    passages = list(dict.fromkeys(doc.page_content for doc in docs))

    if reranker is not None and passages:
        reranked = reranker.rerank(question, passages, k=num_docs_final)
        return [str(item["content"]) for item in reranked]
    return passages[:num_docs_final]


def answer_with_tourism_rag(
    question: str,
    reader: ReaderLLM,
    knowledge_index: Any,
    reranker: RAGPretrainedModel | None = None,
    num_retrieved_docs: int = 20,
    num_docs_final: int = 5,
) -> tuple[str, list[str]]:
    """Answer a tourism question using retrieval, optional reranking, and an LLM."""

    passages = retrieve_passages(
        question=question,
        knowledge_index=knowledge_index,
        reranker=reranker,
        num_retrieved_docs=num_retrieved_docs,
        num_docs_final=num_docs_final,
    )
    answer = generate_answer(question=question, passages=passages, reader=reader)
    return answer, passages

