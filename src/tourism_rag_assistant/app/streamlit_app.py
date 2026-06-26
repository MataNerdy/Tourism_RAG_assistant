"""Streamlit demo for the Tourism RAG Assistant."""

from __future__ import annotations

import streamlit as st

from tourism_rag_assistant.llm.generator import build_reader_llm
from tourism_rag_assistant.pipeline import answer_with_tourism_rag
from tourism_rag_assistant.reranking.colbert import load_reranker
from tourism_rag_assistant.retrieval.vector_store import load_vector_store


@st.cache_resource
def load_resources():
    """Load retrieval, reranking, and generation resources once per session."""

    knowledge_index = load_vector_store()
    reranker = load_reranker()
    reader = build_reader_llm()
    return knowledge_index, reranker, reader


st.set_page_config(page_title="Tourism RAG Assistant", layout="wide")
st.title("Tourism RAG Assistant")
st.caption("RAG guide for landmarks: E5 embeddings, Chroma retrieval, ColBERT reranking, and an LLM reader.")

question = st.text_input(
    "Question",
    value="Что посмотреть в Ярославле, если люблю старинные храмы и набережную?",
)

num_docs = st.slider("Final documents after reranking", 1, 8, 5)

if st.button("Ask RAG"):
    knowledge_index, reranker, reader = load_resources()
    answer, docs = answer_with_tourism_rag(
        question=question,
        reader=reader,
        knowledge_index=knowledge_index,
        reranker=reranker,
        num_retrieved_docs=20,
        num_docs_final=num_docs,
    )
    st.subheader("Answer")
    st.write(answer)

    st.subheader("Retrieved context")
    for index, doc in enumerate(docs, start=1):
        with st.expander(f"Document {index}"):
            st.text(doc)

