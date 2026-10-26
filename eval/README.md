# Retrieval evaluation

Does this implementation of PIC retrieve better than plain chunking? This script measures
it on a small, public corpus so anyone can rerun it with their own models.

## Setup

- **Corpus:** the 48 Wikipedia articles of the [SQuAD v1.1](https://rajpurkar.github.io/SQuAD-explorer/)
  dev set, one article per document (paragraphs joined), about 254,000 words in all.
- **Questions:** a seeded sample of the 10,570 SQuAD questions. Every answer is a verbatim
  span of its article, so a chunk is a *hit* when it contains one of the gold answer strings.
- **Methods:** one sentence per chunk; one SQuAD paragraph per chunk; fixed size of 100 words
  without cutting sentences (the paper's baseline); PIC in the paper's setting (no length
  limits). PIC uses the same sentence splitter as the baselines.
- **Retrieval:** every method embeds its chunks with the same embedding model, by default
  `BAAI/bge-large-en-v1.5` run locally (the paper's model); questions are embedded once;
  cosine similarity; Hits@k is the share of questions with a hit among the k nearest chunks.
  The summarizer is whatever `.env` names.

This is a small corpus with one summarizer and one embedding model. The paper measures on
full Wikipedia with other models, so absolute numbers are not comparable; the ranking of
the methods is what this script shows.

## Run

```bash
cp .env.example .env   # set CHUNKGUARD_LLM_MODEL (and a key)
uv run --with numpy --with sentence-transformers python eval/run.py --env .env --questions 300
```

`--embed env` uses the `.env` embedding model through LiteLLM instead of the local one;
free tiers tend to be too tight for the ~19,000 texts involved. `--llm-pause` spaces the
48 summary calls (default 6 s).

Summaries and embeddings are cached under `eval/cache/` (git-ignored); a rerun with other
settings costs nothing. Results land in `eval/results.md` and `eval/results.json`.

## Results

Preliminary: 11 of the 48 documents (the Gemini free tier allows 20 LLM requests per day
and model), 300 questions drawn from them. The two word-budget columns score the nearest
chunks that fit into 200 or 500 retrieved words, which removes the advantage long chunks
have under Hits@k.

| Method | Chunks | Mean words | Hits@1 | Hits@5 | Hits@20 | Hits@200 words | Hits@500 words |
|---|---|---|---|---|---|---|---|
| Sentence | 2486 | 23.7 | 63.7 | 86.7 | 95.7 | 90.0 | 95.0 |
| Paragraph | 574 | 102.4 | 71.3 | 94.3 | 99.0 | 77.0 | 92.3 |
| Fixed-size (100 words) | 687 | 85.6 | 65.3 | 89.7 | 98.0 | 82.0 | 90.3 |
| PIC (paper: no length limits) | 1078 | 54.5 | 64.0 | 87.7 | 95.7 | 72.0 | 84.0 |
| PIC, merged to ≥ 50 words | 516 | 113.9 | 66.0 | 88.3 | 97.7 | 68.3 | 84.0 |
| PIC, 100-word cap, merged ≥ 50 | 814 | 72.2 | 71.3 | 90.3 | 98.3 | 83.0 | 91.3 |

300 questions (seed 0), 11 documents, LLM `gemini/gemini-3.5-flash`, embeddings `BAAI/bge-large-en-v1.5`. PIC marked 50.2% of its chunks relevant; 69.4% of the chunks retrieved in the top 5 were relevant ones.

How to read this, honestly:

- **PIC in the paper's setting does not beat fixed-size chunking here.** It lands at
  sentence level under Hits@k and below every baseline under a word budget.
- **PIC with a 100-word cap and merging to at least 50 words** (the `max_length`/`min_length`
  options) is on par with or slightly ahead of fixed-size on every column, with the best
  Hits@1 of all methods. The differences are small: with 300 questions the standard error
  is about 2 points, so gaps under 4 points are noise.
- **Each metric favours a chunk size.** Hits@k rewards long chunks (Paragraph wins), the
  word budget rewards short ones (Sentence wins). PIC's relevance flag works as the paper
  describes: half of its chunks are flagged relevant, but 69% of the chunks retrieved in
  the top 5 are relevant ones.
- **SQuAD favours the paragraph baseline:** its questions were written per paragraph.

Rerun with `--questions 300` once all 48 summaries are cached to replace this table.
