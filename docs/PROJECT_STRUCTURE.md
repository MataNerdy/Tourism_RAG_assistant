# Project Structure

```text
Tourism_RAG_assistant/
|
|-- README.md
|-- LICENSE
|-- Makefile
|-- pyproject.toml
|-- requirements.txt
|-- config/
|   `-- default.yaml
|-- data/
|   `-- README.md
|-- docs/
|   |-- ARCHITECTURE.md
|   `-- PROJECT_STRUCTURE.md
|-- examples/
|   `-- sample_query.md
|-- images/
|   |-- data_cleaning_funnel.png
|   |-- data_distribution.png
|   |-- pca.png
|   |-- pics.png
|   |-- pipeline.png
|   |-- pipeline_rag.png
|   |-- rag_metrics.png
|   |-- ragas.png
|   `-- umap.png
|-- notebooks/
|   |-- RAG.ipynb
|   `-- RAG_original.ipynb
|-- src/
|   `-- tourism_rag_assistant/
|       |-- config.py
|       |-- pipeline.py
|       |-- app/
|       |-- evaluation/
|       |-- ingestion/
|       |-- llm/
|       |-- reranking/
|       |-- retrieval/
|       |-- utils/
|       `-- visualization/
`-- tests/
    `-- test_preprocessing.py
```

## Directory Responsibilities

`config/` documents default paths, model names, and preprocessing parameters. The importable runtime config is implemented in `src/tourism_rag_assistant/config.py`.

`data/` is a placeholder for runtime data. Raw downloads and processed datasets are ignored to keep the repository lightweight.

`docs/` contains engineering documentation that explains architectural decisions separately from the quick-start README.

`examples/` contains reproducible usage notes and sample queries.

`images/` stores project diagrams and evaluation visualizations generated during the original analysis.

`notebooks/` keeps the exploratory workflow for transparency. Production-style logic has been moved into Python modules.

`src/tourism_rag_assistant/` is the importable package:

- `ingestion/` downloads, cleans, filters, and aggregates the dataset.
- `retrieval/` converts rows into LangChain documents and persists a Chroma vector store.
- `reranking/` loads the ColBERTv2 reranker.
- `llm/` builds the reader prompt and text-generation pipeline.
- `evaluation/` provides lightweight RAG quality metrics.
- `visualization/` creates PCA and UMAP embedding-space reports.
- `app/` contains the Streamlit demo.

`tests/` contains focused unit tests for deterministic preprocessing helpers.

