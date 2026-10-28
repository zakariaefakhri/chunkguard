# Changelog

## 0.4.0 (2026-10-28)

- Fix: 0.3.0's sentence rules dropped 562 of 13,889 English sentence boundaries on the UD
  English-EWT treebank (recall 70.6 instead of sentencex's 74.6). The lowercase rule now only
  repairs what it was meant for, a short abbreviation before a non-ASCII lowercase word
  ("Abb. über"); the ordinal rule only applies to languages that write ordinals with a
  period (German, Danish, Hungarian, Czech, Polish, ...), so "the age of 13. Police said"
  ends a sentence again. English recall is back to sentencex's, German unchanged.
- `abbreviations=` on `PICChunker` and `split_sentences` adds your own abbreviations.
- `eval/sentences.py` measures sentence boundaries on UD German-GSD and English-EWT.
  Abbreviations learned with Punkt did not help there and are not part of the library.

## 0.3.0 (2026-10-28)

Breaking:

- `chunker_from_env()` and `async_chunker_from_env()` no longer read `.env` from the working
  directory. Pass the path (`chunker_from_env(".env")`) or set the `CHUNKGUARD_*` variables in
  the environment. A path that does not exist raises `FileNotFoundError`. Importing the
  LiteLLM backend also sets `LITELLM_MODE=PRODUCTION` if unset, which stops LiteLLM from
  loading a `.env` of its own on import.

Fixes and improvements (from a code review):

- Sentence splitting: "Fa. Rewe" is no longer split; "Tel. 069 123." and "Hauptstr. 5." end
  their sentence again (the earlier ordinal rule merged them with the next one). Ordinals count
  as such only after a lowercase word, at a sentence start or after a sentence end
  ("im 18. Jahrhundert", "Der 3. Platz"); German and English abbreviations sentencex misses
  are handled until it covers them.
- Excerpts for long documents consist of whole sentences.
- The async embedder sends its batches in parallel, at most `EMBEDDING_CONCURRENCY` (default 4)
  at a time.

## 0.2.1 (2026-10-26)

- The `litellm` extra now installs `tenacity`, which LiteLLM needs for `num_retries` but does
  not declare. Without it, the first provider error raised "tenacity import failed" instead
  of retrying.

## 0.2.0 (2026-10-26)

- `split(..., summary=...)` reuses a stored summary and skips the LLM call, for re-chunking
  with other limits and for reproducible boundaries.
- Limits can be measured in tokens: pass `length=token_length()` (extra `chunkguard[tokens]`,
  tiktoken) or any `str -> int` function. `max_chars`/`min_chars` are renamed to
  `max_length`/`min_length`; the old names still work with a `DeprecationWarning`.
  `.env`: `CHUNKGUARD_LENGTH_UNIT=chars|tokens`, `CHUNKGUARD_TOKEN_ENCODING`,
  `MAX_CHUNK_LENGTH`/`MIN_CHUNK_LENGTH` (the `*_CHARS` names are still read).
- `Chunk.language` carries the language passed to `split`.
- A chunk's length is the sum of its sentences; whitespace between sentences no longer counts.

## 0.1.1 (2026-10-26)

- Merging short chunks is off by default (`min_chars=0`); with 200, a four-sentence document
  came back as one chunk flagged irrelevant.
- An LLM function returning `None` raises the "empty summary" error instead of an
  `AttributeError`.
- `chunker_from_env` without the required settings names the missing `CHUNKGUARD_*` variables.

## 0.1.0 (2026-10-26)

First release: PIC chunking with any LLM and embedding model, LiteLLM extra for `.env`
configuration, sentencex sentence splitting, async variant.
