.PHONY: install prepare index app evaluate-viz test

install:
	pip install -r requirements.txt
	pip install -e .

prepare:
	python -m tourism_rag_assistant.ingestion.preprocessing

index:
	python -m tourism_rag_assistant.retrieval.vector_store

evaluate-viz:
	python -m tourism_rag_assistant.visualization.embedding_space

app:
	streamlit run src/tourism_rag_assistant/app/streamlit_app.py

test:
	pytest -q

