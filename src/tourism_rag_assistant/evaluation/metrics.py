"""Lightweight evaluation helpers for the RAG pipeline."""

from __future__ import annotations

import json
import os
import re
from typing import Any

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from tqdm.auto import tqdm

from tourism_rag_assistant.config import CONFIG
from tourism_rag_assistant.llm.generator import ReaderLLM
from tourism_rag_assistant.pipeline import answer_with_tourism_rag


FALLBACK_FOLLOWUP_QUESTIONS = [
    "О чём говорится в тексте?",
    "Где находится описанный объект?",
    "Чем он может быть интересен туристу?",
]


def generate_followup_questions(answer: str, llm: Any, n: int = 3) -> list[str]:
    """Ask the LLM to generate follow-up questions for answer relevancy scoring."""

    prompt = f"""
Ты ассистент, который придумывает уточняющие вопросы по тексту.
Сформулируй ровно {n} вопросов.

Формат ответа строго JSON:
{{"questions": ["...", "...", "..."]}}

Ответ:
\"\"\"{answer}\"\"\"
""".strip()
    output = llm(prompt, max_new_tokens=200, do_sample=True, temperature=0.6, top_p=0.9)
    text = output[0]["generated_text"] if isinstance(output, list) else output["generated_text"]

    try:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        questions = json.loads(match.group(0)).get("questions", []) if match else []
    except (json.JSONDecodeError, AttributeError):
        questions = []

    return (questions or FALLBACK_FOLLOWUP_QUESTIONS)[:n]


def build_eval_dataset(
    df: pd.DataFrame,
    reader: ReaderLLM,
    knowledge_index: Any,
    n_samples: int = 100,
    save_path: str = str(CONFIG.eval_data_path),
) -> pd.DataFrame:
    """Generate a cached evaluation dataset from sampled place rows."""

    df_eval = df.sample(n_samples, random_state=CONFIG.random_state).copy()
    df_eval["question"] = df_eval.apply(
        lambda row: f"Что такое {row['Name']} в городе {row['City']}?",
        axis=1,
    )

    output_df = pd.read_parquet(save_path) if os.path.exists(save_path) else pd.DataFrame()

    for _, row in tqdm(df_eval.iloc[len(output_df) :].iterrows(), total=n_samples - len(output_df)):
        answer, contexts = answer_with_tourism_rag(
            question=row["question"],
            reader=reader,
            knowledge_index=knowledge_index,
            num_retrieved_docs=5,
            num_docs_final=3,
        )
        output_row = row.to_dict()
        output_row["answer"] = answer
        output_row["contexts"] = contexts
        output_row["ground_truths"] = [row["description"]]
        output_row["followup_questions"] = generate_followup_questions(answer, reader.pipeline)
        output_df = pd.concat([output_df, pd.DataFrame([output_row])], ignore_index=True)
        output_df.to_parquet(save_path, index=False)

    return output_df


def answer_relevancy(
    df_eval: pd.DataFrame,
    model: SentenceTransformer | None = None,
    model_name: str = CONFIG.embedding_model,
) -> tuple[float, list[float]]:
    """Estimate answer relevancy from generated follow-up questions."""

    if model is None:
        model = SentenceTransformer(model_name)

    answers = df_eval["answer"].astype(str).tolist()
    followups = df_eval["followup_questions"].tolist()
    answer_embeddings = model.encode(answers, normalize_embeddings=True)

    scores: list[float] = []
    for index, questions in enumerate(followups):
        if not isinstance(questions, list) or not questions:
            scores.append(0.0)
            continue
        question_embeddings = model.encode(questions, normalize_embeddings=True)
        similarities = question_embeddings @ answer_embeddings[index]
        scores.append(float(np.mean(similarities)))
    return float(np.mean(scores)), scores


def is_relevant_context(row: pd.Series, context: str) -> bool:
    """Check whether a context contains both place and city token overlap."""

    name_tokens = set(str(row["Name"]).lower().split())
    city_tokens = set(str(row["City"]).lower().split())
    context_tokens = set(str(context).lower().split())
    return bool(name_tokens & context_tokens) and bool(city_tokens & context_tokens)


def context_recall(df_eval: pd.DataFrame) -> tuple[float, list[float]]:
    """Measure whether at least one retrieved context matches the target place."""

    scores: list[float] = []
    for _, row in df_eval.iterrows():
        contexts = row["contexts"]
        flags = [is_relevant_context(row, ctx) for ctx in contexts] if isinstance(contexts, list) else []
        scores.append(1.0 if any(flags) else 0.0)
    return float(np.mean(scores)), scores


def context_precision(df_eval: pd.DataFrame) -> tuple[float, list[float]]:
    """Measure the fraction of retrieved contexts matching the target place."""

    scores: list[float] = []
    for _, row in df_eval.iterrows():
        contexts = row["contexts"]
        if not isinstance(contexts, list) or not contexts:
            scores.append(0.0)
            continue
        flags = [is_relevant_context(row, ctx) for ctx in contexts]
        scores.append(float(np.mean(flags)))
    return float(np.mean(scores)), scores

