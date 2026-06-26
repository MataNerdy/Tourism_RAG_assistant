"""ColBERT reranker adapter."""

from __future__ import annotations

from ragatouille import RAGPretrainedModel

from tourism_rag_assistant.config import CONFIG


def load_reranker() -> RAGPretrainedModel:
    """Load the pretrained ColBERTv2 reranker."""

    return RAGPretrainedModel.from_pretrained(CONFIG.reranker_model)

