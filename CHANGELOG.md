# Changelog

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
