# Data

This directory is intentionally lightweight.

The raw dataset is downloaded by:

```bash
make prepare
```

Runtime data is ignored by Git:

- `data/raw/` stores the downloaded source CSV.
- `data/processed/tourism_rag.csv` stores the cleaned retrieval corpus.
- `chroma_db/` stores the persisted Chroma vector index.

The project documentation reports only numbers already observed in the original notebook and README. Re-run the preprocessing pipeline before comparing new results.

