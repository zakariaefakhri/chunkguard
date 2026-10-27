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

37 of the 48 documents, 300 questions drawn from them. Summaries by `gpt-oss-120b` on Groq;
because Groq's free tier allows 8,000 tokens per request, documents longer than 26,000
characters (about half) reached the summarizer as eight evenly spread excerpts, which the
paper does not do. The remaining 11 documents are pending (Groq's daily token limit).

The two word-budget columns score the nearest chunks that fit into 200 or 500 retrieved
words. They remove the advantage long chunks have under Hits@k.

| Method | Chunks | Mean words | Hits@1 | Hits@5 | Hits@20 | Hits@200 words | Hits@500 words |
|---|---|---|---|---|---|---|---|
| Sentence | 7834 | 25.1 | 68.7 | 86.3 | 96.3 | 88.3 | 95.3 |
| Paragraph | 1592 | 123.3 | 76.0 | 94.3 | 98.7 | 79.7 | 92.0 |
| Fixed-size (100 words) | 2322 | 84.6 | 69.7 | 90.0 | 96.3 | 78.3 | 90.7 |
| PIC (paper: no length limits) | 3331 | 58.9 | 66.7 | 90.3 | 96.7 | 74.0 | 88.0 |
| PIC, merged to ≥ 50 words | 1686 | 116.4 | 64.0 | 89.0 | 96.7 | 66.7 | 85.0 |
| PIC, 100-word cap, merged ≥ 50 | 2673 | 73.4 | 71.3 | 90.0 | 95.7 | 83.0 | 91.0 |

300 questions (seed 0), 37 documents, LLM `groq/openai/gpt-oss-120b`, embeddings `BAAI/bge-large-en-v1.5`, summary input capped at 26,000 characters. PIC marked 50.3% of its chunks relevant; 67.7% of the chunks retrieved in the top 5 were relevant ones.

How to read this:

- **PIC in the paper's setting does not beat fixed-size chunking here.** It ties at Hits@5
  and Hits@20 and trails at Hits@1 and under both word budgets.
- **PIC with a 100-word cap and merging to at least 50 words** (the `max_length`/`min_length`
  options) ties fixed-size at Hits@5/20 and is ahead at Hits@1 (+1.6) and at a 200-word
  budget (+4.7). With 300 questions one standard error is about 2.3 points, so only the
  200-word gap is more than noise, and only just.
- **Each metric favours a chunk size.** Hits@k rewards long chunks, where Paragraph wins; it
  is also helped by SQuAD's questions having been written per paragraph. The word budget
  rewards short chunks, where Sentence wins.
- **The relevance flag works as the paper describes:** half of PIC's chunks are flagged
  relevant, and 68% of the chunks retrieved in the top 5 are relevant ones.
- An earlier run on 11 documents with Gemini 3.5 Flash as summarizer showed the same
  pattern.
