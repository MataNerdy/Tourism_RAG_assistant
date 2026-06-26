"""Backward-compatible wrapper for the original notebook RAG API."""

from __future__ import annotations

from typing import Any

from transformers import Pipeline

from tourism_rag_assistant.llm.generator import ReaderLLM, build_reader_llm as _build_reader_llm
from tourism_rag_assistant.pipeline import answer_with_tourism_rag as _answer_with_tourism_rag
from tourism_rag_assistant.reranking.colbert import load_reranker


def build_reader_llm() -> tuple[Pipeline, str]:
    """Return the original `(pipeline, prompt_template)` pair."""

    reader = _build_reader_llm()
    return reader.pipeline, reader.prompt_template


def answer_with_tourism_rag(
    question: str,
    llm: Pipeline,
    prompt_template: str,
    knowledge_index: Any,
    reranker: Any | None = None,
    num_retrieved_docs: int = 20,
    num_docs_final: int = 5,
) -> tuple[str, list[str]]:
    """Run the package pipeline through the original function signature."""

    reader = ReaderLLM(pipeline=llm, prompt_template=prompt_template)
    return _answer_with_tourism_rag(
        question=question,
        reader=reader,
        knowledge_index=knowledge_index,
        reranker=reranker,
        num_retrieved_docs=num_retrieved_docs,
        num_docs_final=num_docs_final,
    )


__all__ = ["answer_with_tourism_rag", "build_reader_llm", "load_reranker"]

