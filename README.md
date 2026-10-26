# chunkguard

Document chunking for RAG with **PIC** (Pseudo-Instruction for document Chunking,
Wang et al., ACL 2025).

When the knowledge base is built, the user queries are unknown. PIC stands in for them
with an LLM summary of the document. Every sentence is embedded and compared with the
summary by cosine similarity; the threshold is the mean of those similarities. Neighbouring
sentences above it form a chunk of core content, neighbouring sentences below it a chunk of
side details. Chunk boundaries sit where the text turns towards or away from the topic.
Order is kept, no sentence is cut, nothing is dropped.

Cost: one LLM call per document at indexing time, plus the embeddings. No training.

## Use

```python
from chunkguard import chunker_from_env

chunker = chunker_from_env()  # reads .env
chunks = chunker.split(doc_id="doc-1", text=cleaned_text, language="de")
```

Each `Chunk` carries `text`, `start`/`end` (character offsets into the input),
`sentence_start`/`sentence_end` (sentence range, end exclusive), `relevant`,
`similarity` and `summary`.

The chunker is an adapter: text in, chunks out. Extraction before it and indexing after it
do not depend on how it splits. For other backends or tests, pass any two callables:

```python
PICChunker(summarize=lambda text, language: ..., embed=lambda texts: ...)
```

## Steps

1. Split into sentences (rule-based, abbreviations for de/en/it/hu; line breaks always end a sentence).
2. Summarize the document with the LLM.
3. Embed summary and sentences; cosine similarity of each sentence to the summary.
4. Threshold = mean similarity; a sentence is relevant if its similarity is at or above it.
5. Group runs of neighbouring sentences with the same flag.
6. Split chunks longer than `MAX_CHUNK_CHARS` at sentence boundaries into near-equal pieces.
7. Merge chunks shorter than `MIN_CHUNK_CHARS` into the neighbour with the closest mean
   similarity, unless that would exceed the maximum. A merged chunk is `relevant` when its
   relevant sentences make up at least half of its text.

## Configuration

Copy `.env.example` to `.env`. LLM and embedding model are configured separately, each as
`openai` (with `BASE_URL` for any OpenAI-compatible endpoint) or `azure` (model = deployment
name, `BASE_URL` = Azure endpoint, `API_VERSION` required).

## Development

```bash
uv sync
uv run pytest
```
