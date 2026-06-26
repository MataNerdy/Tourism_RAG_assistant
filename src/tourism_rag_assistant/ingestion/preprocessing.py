"""Prepare the tourism dataset for retrieval.

The source data contains multiple image captions and metadata rows for the same
tourism object. This module normalizes text, filters obvious noise, checks
caption-description consistency, and aggregates rows into one retrieval document
per landmark.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from tourism_rag_assistant.config import CONFIG
from tourism_rag_assistant.utils.text import has_suspect_text, normalize_text


def download_raw_data(output_path: Path = CONFIG.raw_data_path) -> Path:
    """Download the source CSV if it is not present locally."""

    import gdown

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        return output_path
    gdown.download(CONFIG.raw_data_url, str(output_path), quiet=False)
    return output_path


def add_retrieval_text_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Create normalized helper fields used for filtering and grouping."""

    df = df.copy()
    for column in ["description", "en_txt", "Name", "City"]:
        df[column] = df[column].map(normalize_text)

    df["place_hints"] = (df["Name"] + " " + df["City"]).map(normalize_text)
    df["text"] = (
        df["Name"] + " " + df["City"] + " " + df["description"] + " " + df["en_txt"]
    ).map(normalize_text)
    return df


make_text_columns = add_retrieval_text_columns


def pick_best_name(variants: Iterable[str]) -> str:
    """Select the most descriptive normalized name variant for a place."""

    unique_variants = [variant for variant in set(variants) if isinstance(variant, str)]
    if not unique_variants:
        return ""
    return max(unique_variants, key=len)


def select_typical_text_indices(texts: list[str], k: int = 3) -> list[int]:
    """Pick captions closest to the TF-IDF centroid of a place group."""

    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    if len(texts) <= k:
        return list(range(len(texts)))

    vectorizer = TfidfVectorizer(
        max_features=30_000,
        token_pattern=r"(?u)\b[\w\-]{2,}\b",
        lowercase=True,
    )
    matrix = vectorizer.fit_transform(texts)
    centroid = np.asarray(matrix.mean(axis=0)).ravel().reshape(1, -1)
    similarities = cosine_similarity(matrix, centroid).ravel()

    selected: list[int] = []
    used_texts: set[str] = set()
    for idx in np.argsort(-similarities):
        if texts[idx] in used_texts:
            continue
        selected.append(int(idx))
        used_texts.add(texts[idx])
        if len(selected) >= k:
            break
    return selected


topk_typical_texts = select_typical_text_indices


def clean_caption(text: object) -> str:
    """Clean a generated image caption without changing its meaning."""

    if not isinstance(text, str):
        text = str(text)
    text = re.sub(r"\barafed\b", "", text)
    return re.sub(r"\s+", " ", text).strip()


def build_final_caption_text(group: pd.DataFrame, k: int = 3, max_chars: int = 2500) -> str:
    """Aggregate representative captions for one landmark."""

    texts = group["en_txt"].fillna("").astype(str).tolist()
    selected_indices = select_typical_text_indices(texts, k=k)
    selected_texts = [texts[index] for index in selected_indices if texts[index].strip()]
    final_text = " ".join(selected_texts)
    return clean_caption(final_text[:max_chars])


build_text_final = build_final_caption_text
flag_suspect_text = has_suspect_text


def balanced_sample_by_city(
    df: pd.DataFrame,
    target_n: int = CONFIG.target_n_places,
    random_state: int = CONFIG.random_state,
) -> pd.DataFrame:
    """Sample a city-balanced subset for the final retrieval corpus."""

    rng = np.random.default_rng(random_state)
    groups = list(df.groupby("City", dropna=False))
    base_quota = max(1, target_n // max(1, len(groups)))

    selected_indices: list[int] = []
    for _, group in groups:
        n = min(base_quota, len(group))
        selected_indices.extend(group.sample(n=n, random_state=int(rng.integers(1e9))).index)

    if len(selected_indices) < target_n:
        remaining = df.drop(index=selected_indices).sort_values(
            ["n_records_used", "n_records_total"], ascending=False
        )
        selected_indices.extend(remaining.head(target_n - len(selected_indices)).index)

    return df.loc[selected_indices].head(target_n).reset_index(drop=True)


def prepare_dataset(
    raw_path: Path = CONFIG.raw_data_path,
    output_path: Path = CONFIG.processed_data_path,
) -> pd.DataFrame:
    """Build the cleaned, aggregated retrieval dataset."""

    if not raw_path.exists():
        download_raw_data(raw_path)

    raw = pd.read_csv(raw_path, encoding="utf-8-sig")
    df = raw.drop(columns=["Unnamed: 0"], errors="ignore")
    df = df.dropna(subset=["description", "WikiData"]).reset_index(drop=True)
    df = add_retrieval_text_columns(df)

    df = df.drop_duplicates(subset=["en_txt"]).reset_index(drop=True)

    canonical_names = (
        df.groupby("WikiData")["place_hints"]
        .apply(pick_best_name)
        .reset_index(name="canonical_place_name")
    )
    df = df.merge(canonical_names, on="WikiData", how="right")
    df["place_hints"] = df["canonical_place_name"]

    df["is_suspect_regex"] = df["text"].apply(has_suspect_text)
    df = df[~df["is_suspect_regex"]].copy()

    from sentence_transformers import SentenceTransformer

    cleaning_model = SentenceTransformer(CONFIG.cleaning_model)
    description_embeddings = cleaning_model.encode(
        df["description"].tolist(),
        batch_size=128,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    caption_embeddings = cleaning_model.encode(
        df["en_txt"].tolist(),
        batch_size=128,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    df["similarity"] = (description_embeddings * caption_embeddings).sum(axis=1)

    good_caption_mask = (
        (df["en_txt"].str.len() >= CONFIG.min_caption_len)
        & (df["similarity"] >= CONFIG.similarity_threshold)
    )
    df_good = df[good_caption_mask].copy()

    rows: list[dict[str, object]] = []
    for place, group_all in df.groupby("place_hints", dropna=False):
        group_good = df_good.loc[group_all.index.intersection(df_good.index)]
        source_group = group_good if len(group_good) > 0 else group_all
        first_row = group_all.iloc[0]

        rows.append(
            {
                "place_hints": place,
                "WikiData": first_row.get("WikiData", ""),
                "Name": first_row.get("Name", ""),
                "City": first_row.get("City", ""),
                "Lon": first_row.get("Lon", np.nan),
                "Lat": first_row.get("Lat", np.nan),
                "description": first_row.get("description", ""),
                "image": first_row.get("image", ""),
                "text_final_clean": build_final_caption_text(
                    source_group,
                    k=CONFIG.topk_per_place,
                    max_chars=CONFIG.max_final_chars,
                ),
                "n_records_total": len(group_all),
                "n_records_used": len(source_group),
                "sim_mean_used": float(source_group["similarity"].mean()),
                "sim_min_used": float(source_group["similarity"].min()),
            }
        )

    places = pd.DataFrame(rows)
    retrieval_dataset = balanced_sample_by_city(places)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    retrieval_dataset.to_csv(output_path, index=False)
    return retrieval_dataset


if __name__ == "__main__":
    df_rag = prepare_dataset()
    print(f"Prepared {len(df_rag)} RAG documents")
