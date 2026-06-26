"""Chroma vector store construction and loading."""

from __future__ import annotations

import pandas as pd
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

from tourism_rag_assistant.config import CONFIG


def row_to_document(row: pd.Series) -> Document:
    """Convert one aggregated landmark row into a retrievable document."""

    page_content = "\n".join(
        [
            f"Название: {row['Name']}",
            f"Город: {row['City']}",
            "",
            "Описание (WikiData):",
            str(row["description"]),
            "",
            "Описание по фото:",
            str(row["text_final_clean"]),
        ]
    )
    metadata = {
        "place_hints": row["place_hints"],
        "wikidata_id": row["WikiData"],
        "city": row["City"],
        "image_base64": row.get("image", ""),
    }
    return Document(page_content=page_content, metadata=metadata)


def build_embedding_function() -> HuggingFaceEmbeddings:
    """Create the E5 embedding function used by Chroma."""

    return HuggingFaceEmbeddings(
        model_name=CONFIG.embedding_model,
        model_kwargs={"device": CONFIG.model_device},
        encode_kwargs={"normalize_embeddings": True},
    )


def build_vector_store(df: pd.DataFrame) -> Chroma:
    """Create and persist a Chroma index from the processed dataset."""

    documents = [row_to_document(row) for _, row in df.iterrows()]
    return Chroma.from_documents(
        documents,
        build_embedding_function(),
        collection_name=CONFIG.collection_name,
        persist_directory=str(CONFIG.chroma_dir),
    )


def load_vector_store() -> Chroma:
    """Load the persisted Chroma collection."""

    return Chroma(
        collection_name=CONFIG.collection_name,
        persist_directory=str(CONFIG.chroma_dir),
        embedding_function=build_embedding_function(),
    )


if __name__ == "__main__":
    df_rag = pd.read_csv(CONFIG.processed_data_path)
    build_vector_store(df_rag)
    print(f"Vector store saved to {CONFIG.chroma_dir}")

