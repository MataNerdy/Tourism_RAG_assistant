# Sample Query

## Question

Что посмотреть в Ярославле, если люблю старинные храмы и набережную?

## Expected RAG behavior

1. Retrieve documents about Yaroslavl.
2. Prefer churches, monasteries, historic buildings, and embankment-related landmarks.
3. Rerank retrieved candidates with ColBERT.
4. Answer only from retrieved context and make the scope of the recommendation clear.

## Observed recommendation pattern

The original notebook run recommended historic churches, the Tolga Monastery, the Volga embankment, and the historic city center based on retrieved and reranked context.

