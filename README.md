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
pip install chunkguard              # core: bring your own model functions (Pydantic + sentencex)
pip install "chunkguard[litellm]"   # plus configuration by environment for 100+ providers via LiteLLM
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

Options: `max_length` (default 2000), `min_length` (default 0), `length` (default `len`, see
[Token-based limits](#token-based-limits)), `summary_max_chars`, `summary_prompt`,
`abbreviations`, `sentence_splitter`.

**Option 2: `.env` with LiteLLM.** Install the extra, copy [`.env.example`](.env.example) to
`.env` and set the two models as [LiteLLM strings](https://docs.litellm.ai/docs/providers):

```bash
CHUNKGUARD_LLM_MODEL=anthropic/claude-sonnet-5-5
CHUNKGUARD_EMBEDDING_MODEL=cohere/embed-multilingual-v3.0
```

Put keys in `.env` as `CHUNKGUARD_LLM_API_KEY` / `CHUNKGUARD_EMBEDDING_API_KEY`. If those are
unset, LiteLLM uses the provider's usual variables (`ANTHROPIC_API_KEY`, `COHERE_API_KEY`,
`AWS_*`, ...) from the process environment. Azure, Bedrock, Vertex, Ollama, vLLM and any
OpenAI-compatible server work the same way. The `.env` file is read only when you pass its
path; without one, the `CHUNKGUARD_*` variables come from the process environment. Then:

```python
from chunkguard import chunker_from_env

chunker = chunker_from_env(".env")
chunks = chunker.split(doc_id="report-7", text=cleaned_text, language="en")

for chunk in chunks:
    print(chunk.index, chunk.relevant, round(chunk.similarity, 3), chunk.text[:60])
```

`split` takes plain text that extraction has already cleaned (line breaks inside sentences
joined). Each `Chunk` carries:

| Field | Meaning |
|---|---|
| `doc_id`, `index` | Document and position of the chunk in it |
| `language` | The language passed to `split` |
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

chunker = async_chunker_from_env(".env")
chunks = await chunker.split("report-7", cleaned_text, "en")
```

`AsyncPICChunker` takes async `complete` and `embed` functions in the same way.

### Reusing the summary

Every `split` call asks the LLM for a summary. Pass one you stored to skip that call, for
example to re-chunk with other limits, or to keep boundaries reproducible (LLM output varies
between calls):

```python
chunks = chunker.split("report-7", cleaned_text, "en", summary=chunks[0].summary)
```

### Token-based limits

`max_length` and `min_length` are measured by `length`, which is `len` (characters) by
default. Embedding models limit tokens, so you can count those instead:

```python
from chunkguard import PICChunker, token_length  # pip install "chunkguard[tokens]"

chunker = PICChunker(complete, embed, max_length=512, length=token_length("o200k_base"))
```

Any `str -> int` function works, for example your embedding model's own tokenizer. tiktoken
downloads the encoding on first use.

### Your own summary prompt

The default is the paper's prompt (`chunkguard.SUMMARY_PROMPT`) plus an instruction to answer
in the document's language. Pass `summary_prompt` with `{text}` and `{language}` placeholders
to change it.

### Your own abbreviations

Domain texts use abbreviations no general list knows ("Flst.", "lfd.", "Bauabschn."). Pass them
so they never end a sentence (case-insensitive, with or without the period):

```python
chunker = PICChunker(complete, embed, abbreviations={"Flst", "lfd", "Bauabschn"})
```

### Your own sentence splitter

Sentences come from [sentencex](https://github.com/wikimedia/sentencex) (see [Limits](#limits)).
To use another splitter, pass a function
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

1. **Sentences.** Split into sentences with sentencex, keeping character offsets. Every line
   break also ends a sentence, so headings and list items stand alone. chunkguard drops
   sentencex boundaries before a lowercase letter, after abbreviations it misses ("Fa.",
   "Tel.", "inkl.", "Hauptstr."; German and English, measured) and after ordinals ("im 18.
   Jahrhundert"), but keeps them after house, phone and page numbers ("Hauptstr. 5.").
2. **Summary.** One LLM call with the summarizer prompt from the paper (Appendix C), plus an
   instruction to answer in the document's language.
3. **Similarity.** Embed the summary and every sentence; cosine similarity of each sentence
   to the summary.
4. **Threshold.** The mean of those similarities. A sentence is relevant at or above it.
5. **Grouping.** Each run of neighbouring sentences with the same flag becomes one chunk.

Steps 1 to 5 are the method as published. chunkguard adds three things for production use:

6. **Long chunks** (over `max_length`, default 2000 characters) are split at sentence
   boundaries into near-equal pieces. `max_length=None` switches this off.
7. **Short chunks** merge into the neighbour with the closest mean similarity when you set
   `min_length` (off by default), unless that would exceed `max_length`. A merged chunk is
   `relevant` when its relevant sentences make up at least half of its text. Note that with
   merging, a short document can end up as a single chunk.
8. **Long documents** reach the summarizer as eight evenly spread runs of whole sentences totalling
   `summary_max_chars`, so the call fits the context window and every part of the document
   still shapes the summary.

## Configuration

With plain functions, pass the options to `PICChunker` (`max_length`, `min_length`, `length`,
`summary_max_chars`, `summary_prompt`, `sentence_splitter`). With the LiteLLM extra, they come
from environment variables, prefixed `CHUNKGUARD_`, plus the `.env` file whose path you pass
to `chunker_from_env`:

| Variable | Default | Meaning |
|---|---|---|
| `LLM_MODEL` | required | LiteLLM model string, e.g. `gemini/gemini-2.5-flash` |
| `LLM_API_KEY` | provider variable | API key |
| `LLM_API_BASE` | – | Endpoint, e.g. Azure resource, Ollama or any OpenAI-compatible server |
| `LLM_API_VERSION` | – | Azure only |
| `EMBEDDING_MODEL`, `EMBEDDING_API_*` | | Same settings for the embedding model |
| `EMBEDDING_BATCH_SIZE` | `64` | Texts per embedding request |
| `EMBEDDING_CONCURRENCY` | `4` | Embedding requests in flight at once (async only) |
| `REQUEST_TIMEOUT` | `120` | Seconds per request |
| `MAX_RETRIES` | `2` | Retries per request on provider errors |
| `MAX_CHUNK_LENGTH` | `2000` | Split longer chunks; `0` = no limit |
| `MIN_CHUNK_LENGTH` | `0` | Merge shorter chunks into a neighbour; `0` = off |
| `LENGTH_UNIT` | `chars` | `chars` or `tokens` (tiktoken, extra `tokens`) |
| `TOKEN_ENCODING` | `o200k_base` | tiktoken encoding for `LENGTH_UNIT=tokens` |
| `SUMMARY_MAX_CHARS` | `100000` | Excerpt longer documents for the summary; `0` = send all |

Embedding requests stay below 400,000 characters each. A sentence longer than 8,000
characters is truncated for embedding only; its chunk keeps the full text.

LiteLLM versions 1.82.7 and 1.82.8 were compromised on PyPI in March 2026; the extra requires
1.83.0 or later.

## Limits

- **Sentence splitting is rule-based.** sentencex carries punctuation and abbreviation rules
  for many languages and falls back to related ones; chunkguard's additions (see "How it
  works") are tested for German and English only. On the UD treebanks it matches sentencex
  in English and trails it slightly in German, mostly on faulty gold sentences
  ([eval/sentences.md](eval/sentences.md)). Known misses: "zzgl. MwSt." at the end of a
  sentence is merged with the next one, sentences without final punctuation (headlines) are
  not separated, and languages without sentence punctuation (Thai) are not split. Pass your
  own abbreviations or splitter if that matters.
- **Lengths count sentences, not gaps.** A chunk's length is the sum of its sentences; the
  whitespace between them is not counted, so a chunk can exceed `max_length` by a few
  characters of spacing.
- **The relevance flag is relative to the document.** Nearly every document has relevant and
  irrelevant chunks, however on-topic it is overall. Use it to weight or filter retrieval,
  not as an absolute quality score.
- **Retrieval gains are the paper's results** on English Wikipedia and open-domain QA. Our
  own small measurement ([eval/](eval/)) does not reproduce a gain for the paper's setting;
  see above.

## Does it help?

[eval/](eval/) measures retrieval on Wikipedia articles from the SQuAD dev set, PIC against
sentence, paragraph and fixed-size chunking, with the paper's embedding model
(bge-large-en-v1.5), 37 documents, tuning and test questions kept apart, McNemar's test.

On this benchmark PIC does not beat fixed-size chunking of the same size. PIC in the paper's
setting trails 100-word fixed-size chunks; PIC capped at 60 words beats them, but 60-word
fixed-size chunks do just as well, so the gain comes from the shorter chunks, not from PIC.
Smoothing, other thresholds, summary context in the vector and a bonus for relevant chunks
did not help. SQuAD asks mostly single-sentence factual questions, which is where grouping
related sentences should matter least; datasets with answers spread over several sentences
are still untested. Details in [eval/README.md](eval/README.md).

Measure on your own corpus. If you use chunkguard, prefer short chunks
(`max_length=60, min_length=25, length=lambda t: len(t.split())` scored best here).

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
