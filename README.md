# chunkguard

[![CI](https://github.com/zakariaefakhri/chunkguard/actions/workflows/ci.yml/badge.svg)](https://github.com/zakariaefakhri/chunkguard/actions/workflows/ci.yml)

Document chunking for RAG with **PIC** (Pseudo-Instruction for document Chunking), with any
LLM and any embedding model. From
[*Document Segmentation Matters for Retrieval-Augmented Generation*](https://aclanthology.org/2025.findings-acl.422/)
(Wang et al., Findings of ACL 2025). This is an independent implementation, not the authors' code.

When you build a knowledge base, you don't know the questions users will ask. PIC stands in
for them with an LLM summary of the document. Every sentence is embedded and compared with
the summary; the threshold is the mean similarity. Neighbouring sentences at or above it
form a chunk of core content, neighbouring sentences below it a chunk of side details. So
chunk boundaries sit exactly where the text turns towards or away from its topic, and every
sentence is judged against the whole document rather than just its neighbour, as in
semantic chunking.

Order is kept, no sentence is cut, nothing is dropped. Cost per document: one LLM call and
one embedding pass over its sentences. No training.

## Install

```bash
pip install chunkguard              # core: bring your own model functions, depends only on Pydantic
pip install "chunkguard[litellm]"   # plus .env configuration for 100+ providers via LiteLLM
```

## Choose your models

chunkguard needs two things, and you pick both: an LLM that writes one summary per document,
and an embedding model. They can come from different providers.

**Option 1: plain functions.** `PICChunker` takes any `complete(prompt) -> str` and any
`embed(texts) -> list[vector]`. No provider SDK is involved:

```python
from chunkguard import PICChunker
from sentence_transformers import SentenceTransformer  # local embeddings, as an example
import anthropic  # any LLM, as an example

claude = anthropic.Anthropic()
encoder = SentenceTransformer("BAAI/bge-m3")


def complete(prompt: str) -> str:
    message = claude.messages.create(
        model="claude-sonnet-5-5", max_tokens=512, messages=[{"role": "user", "content": prompt}]
    )
    return message.content[0].text


chunker = PICChunker(complete=complete, embed=lambda texts: encoder.encode(texts).tolist())
chunks = chunker.split(doc_id="report-7", text=cleaned_text, language="en")
```

**Option 2: `.env` with LiteLLM.** Install the extra, copy [`.env.example`](.env.example) to
`.env` and set the two models as [LiteLLM strings](https://docs.litellm.ai/docs/providers):

```bash
CHUNKGUARD_LLM_MODEL=anthropic/claude-sonnet-5-5
CHUNKGUARD_EMBEDDING_MODEL=cohere/embed-multilingual-v3.0
```

Put keys in `.env` as `CHUNKGUARD_LLM_API_KEY` / `CHUNKGUARD_EMBEDDING_API_KEY`. If those are
unset, LiteLLM uses the provider's usual variables (`ANTHROPIC_API_KEY`, `COHERE_API_KEY`,
`AWS_*`, ...) from the process environment. Azure, Bedrock, Vertex, Ollama, vLLM and any
OpenAI-compatible server work the same way. Then:

```python
from chunkguard import chunker_from_env

chunker = chunker_from_env()
chunks = chunker.split(doc_id="report-7", text=cleaned_text, language="en")

for chunk in chunks:
    print(chunk.index, chunk.relevant, round(chunk.similarity, 3), chunk.text[:60])
```

`split` takes plain text that extraction has already cleaned (line breaks inside sentences
joined). Each `Chunk` carries:

| Field | Meaning |
|---|---|
| `doc_id`, `index` | Document and position of the chunk in it |
| `text` | Exact slice `input[start:end]` |
| `start`, `end` | Character offsets into the input (end exclusive) |
| `sentence_start`, `sentence_end` | Sentence range (end exclusive) |
| `relevant` | Core content (`True`) or side details (`False`) |
| `similarity` | Mean cosine similarity of the chunk's sentences to the summary |
| `summary` | The document summary, for use as metadata in the index |

`Chunk` is a Pydantic model; `chunk.model_dump()` gives a dict for your vector store.

### Async

```python
from chunkguard import async_chunker_from_env

chunker = async_chunker_from_env()
chunks = await chunker.split("report-7", cleaned_text, "en")
```

`AsyncPICChunker` takes async `complete` and `embed` functions in the same way.

### Your own summary prompt

The default is the paper's prompt (`chunkguard.SUMMARY_PROMPT`) plus an instruction to answer
in the document's language. Pass `summary_prompt` with `{text}` and `{language}` placeholders
to change it.

### Your own sentence splitter

The built-in splitter is rule-based (see [Limits](#limits)). To use another, pass a function
that returns `(start, end)` character spans. For example, NLTK's Punkt, which the paper uses
for sentence splitting:

```python
from nltk.tokenize.punkt import PunktSentenceTokenizer

punkt = PunktSentenceTokenizer()


def nltk_sentences(text, language):
    return list(punkt.span_tokenize(text))


chunker = PICChunker(complete, embed, sentence_splitter=nltk_sentences)
```

## How it works

1. **Sentences.** Split into sentences, keeping character offsets.
2. **Summary.** One LLM call with the summarizer prompt from the paper (Appendix C), plus an
   instruction to answer in the document's language.
3. **Similarity.** Embed the summary and every sentence; cosine similarity of each sentence
   to the summary.
4. **Threshold.** The mean of those similarities. A sentence is relevant at or above it.
5. **Grouping.** Each run of neighbouring sentences with the same flag becomes one chunk.

Steps 1 to 5 are the method as published. chunkguard adds three things for production use;
set `max_chars=None, min_chars=0, summary_max_chars=None` to switch them off:

6. **Long chunks** (over `max_chars`) are split at sentence boundaries into near-equal pieces.
7. **Short chunks** (under `min_chars`) merge into the neighbour with the closest mean
   similarity, unless that would exceed `max_chars`. A merged chunk is `relevant` when its
   relevant sentences make up at least half of its text.
8. **Long documents** reach the summarizer as eight evenly spread excerpts totalling
   `summary_max_chars`, so the call fits the context window and every part of the document
   still shapes the summary.

## Configuration

With plain functions, pass the options to `PICChunker` (`max_chars`, `min_chars`,
`summary_max_chars`, `summary_prompt`, `sentence_splitter`). With the LiteLLM extra, they come
from environment variables or `.env`, prefixed `CHUNKGUARD_`:

| Variable | Default | Meaning |
|---|---|---|
| `LLM_MODEL` | required | LiteLLM model string, e.g. `gemini/gemini-2.5-flash` |
| `LLM_API_KEY` | provider variable | API key |
| `LLM_API_BASE` | – | Endpoint, e.g. Azure resource, Ollama or any OpenAI-compatible server |
| `LLM_API_VERSION` | – | Azure only |
| `EMBEDDING_MODEL`, `EMBEDDING_API_*` | | Same settings for the embedding model |
| `EMBEDDING_BATCH_SIZE` | `64` | Texts per embedding request |
| `MAX_CHUNK_CHARS` | `2000` | Split longer chunks; `0` = no limit |
| `MIN_CHUNK_CHARS` | `200` | Merge shorter chunks; `0` = no merging |
| `SUMMARY_MAX_CHARS` | `100000` | Excerpt longer documents for the summary; `0` = send all |

Embedding requests stay below 400,000 characters each. A sentence longer than 8,000
characters is truncated for embedding only; its chunk keeps the full text.

LiteLLM versions 1.82.7 and 1.82.8 were compromised on PyPI in March 2026; the extra requires
1.83.0 or later.

## Limits

- **Sentence splitting is rule-based.** It knows abbreviations for German, English, Italian
  and Hungarian, handles full-width punctuation (Chinese, Japanese), `؟` and `।`, and treats
  every line break as a sentence end. It misses some boundaries, such as a sentence ending
  in "etc.", and does not split languages without sentence punctuation (Thai). Pass your own
  splitter if that matters.
- **Sizes are in characters, not tokens.** Pick `MAX_CHUNK_CHARS` with your embedding model's
  token limit in mind.
- **The relevance flag is relative to the document.** Nearly every document has relevant and
  irrelevant chunks, however on-topic it is overall. Use it to weight or filter retrieval,
  not as an absolute quality score.
- **Retrieval gains are the paper's results** on English Wikipedia and open-domain QA. This
  repository does not reproduce them.

## Development

```bash
uv sync
uv run pytest
uv run ruff check . && uv run ruff format --check .
```

Releases: bump `version` in `pyproject.toml`, then push a matching tag (`v0.1.0`). The
release workflow publishes to PyPI via trusted publishing.

## Citation

If you use PIC in research, cite the paper:

```bibtex
@inproceedings{wang-etal-2025-document,
  title     = {Document Segmentation Matters for Retrieval-Augmented Generation},
  author    = {Wang, Zhitong and Gao, Cheng and Xiao, Chaojun and Huang, Yufei and Si, Shuzheng and
               Luo, Kangyang and Bai, Yuzhuo and Li, Wenhao and Duan, Tangjian and Lv, Chuancheng and
               Lu, Guoshan and Chen, Gang and Qi, Fanchao and Sun, Maosong},
  booktitle = {Findings of the Association for Computational Linguistics: ACL 2025},
  pages     = {8063--8075},
  year      = {2025},
  url       = {https://aclanthology.org/2025.findings-acl.422/}
}
```

## License

[MIT](LICENSE)
