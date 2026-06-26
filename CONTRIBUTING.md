# Contributing

Contributions are welcome when they preserve the core project goal: a transparent RAG pipeline for tourism landmark search and grounded answer generation.

## Development Setup

```bash
python -m venv .venv
source .venv/bin/activate
make install
make test
```

## Guidelines

- Keep data artifacts, local vector stores, generated reports, and credentials out of Git.
- Prefer small, reviewable changes with clear motivation.
- Do not introduce new metrics or benchmark claims without reproducible scripts and saved outputs.
- Keep notebooks as exploratory artifacts; reusable logic should live in `src/tourism_rag_assistant/`.
- Add docstrings and type hints for new public functions.
- Update `README.md` and `docs/ARCHITECTURE.md` when changing the pipeline.

## Pull Request Checklist

- Tests pass with `make test`.
- The Streamlit demo still imports successfully.
- Documentation reflects the actual behavior of the code.
- No generated ChromaDB, raw CSV, parquet cache, or `.env` file is committed.

