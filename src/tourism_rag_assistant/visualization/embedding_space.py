"""PCA and UMAP visualizations for persisted document embeddings."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import umap
from plotly.graph_objects import Figure
from sklearn.decomposition import PCA

from tourism_rag_assistant.config import CONFIG
from tourism_rag_assistant.retrieval.vector_store import load_vector_store


def get_embeddings() -> np.ndarray:
    """Read embeddings from the Chroma collection."""

    db = load_vector_store()
    embeddings = db._collection.get(include=["embeddings"])["embeddings"]
    return np.array(embeddings)


def plot_pca(df: pd.DataFrame, output_html: str = "reports/pca_embeddings.html") -> Figure:
    """Write a PCA projection of document embeddings to HTML."""

    embeddings = get_embeddings()
    projection = PCA(n_components=2).fit_transform(embeddings)
    plot_df = pd.DataFrame(
        {
            "pc1": projection[:, 0],
            "pc2": projection[:, 1],
            "city": df["City"].values,
            "place": df["Name"].values,
        }
    )
    fig = px.scatter(
        plot_df,
        x="pc1",
        y="pc2",
        color="city",
        hover_data=["place"],
        title="PCA projection of RAG document embeddings",
    )
    fig.write_html(output_html)
    return fig


def plot_umap(df: pd.DataFrame, output_html: str = "reports/umap_embeddings.html") -> Figure:
    """Write a UMAP projection of document embeddings to HTML."""

    embeddings = get_embeddings()
    projection = umap.UMAP(
        n_neighbors=15,
        min_dist=0.1,
        n_components=2,
        metric="cosine",
        random_state=CONFIG.random_state,
    ).fit_transform(embeddings)
    plot_df = pd.DataFrame(
        {
            "u1": projection[:, 0],
            "u2": projection[:, 1],
            "city": df["City"].values,
            "place": df["Name"].values,
        }
    )
    fig = px.scatter(
        plot_df,
        x="u1",
        y="u2",
        color="city",
        hover_data=["place"],
        title="UMAP projection of RAG document embeddings",
    )
    fig.write_html(output_html)
    return fig


if __name__ == "__main__":
    df_rag = pd.read_csv(CONFIG.processed_data_path)
    plot_pca(df_rag)
    plot_umap(df_rag)

