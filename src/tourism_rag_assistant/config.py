"""Project configuration.

The defaults mirror the original notebook pipeline while keeping paths and
model choices in a single importable object.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class ProjectConfig:
    """Runtime settings for data preparation, retrieval, reranking, and LLM use."""

    raw_data_url: str = "https://drive.google.com/uc?id=1P1BsvI2jPN3fEqjc2YZxmQ-MTs22WVUk"
    raw_data_path: Path = PROJECT_ROOT / "data/raw/file.csv"
    processed_data_path: Path = PROJECT_ROOT / "data/processed/tourism_rag.csv"
    eval_data_path: Path = PROJECT_ROOT / "reports/df_eval_100.parquet"
    chroma_dir: Path = PROJECT_ROOT / "chroma_db/tourism"
    collection_name: str = "tourism_rag"

    embedding_model: str = "intfloat/multilingual-e5-base"
    cleaning_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    reranker_model: str = "colbert-ir/colbertv2.0"
    reader_model: str = "mistralai/Mistral-7B-Instruct-v0.3"
    model_device: str = os.getenv("TOURISM_RAG_DEVICE", "cuda")

    target_n_places: int = 250
    similarity_threshold: float = 0.25
    min_caption_len: int = 20
    topk_per_place: int = 3
    max_final_chars: int = 2500
    random_state: int = 42


CONFIG = ProjectConfig()

