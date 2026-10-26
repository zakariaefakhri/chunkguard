"""PIC document chunking for RAG: text in, chunks out, with any LLM and embedding model."""

from chunkguard.pic import SUMMARY_PROMPT, AsyncPICChunker, Chunk, PICChunker
from chunkguard.sentences import split_sentences
from chunkguard.tokens import token_length

__all__ = [
    "SUMMARY_PROMPT",
    "AsyncPICChunker",
    "Chunk",
    "PICChunker",
    "async_chunker_from_env",
    "chunker_from_env",
    "split_sentences",
    "token_length",
]


def __getattr__(name: str):
    # The .env helpers need the optional litellm extra; import them only on use.
    if name in ("chunker_from_env", "async_chunker_from_env"):
        from chunkguard import backend

        return getattr(backend, name)
    raise AttributeError(f"module 'chunkguard' has no attribute {name!r}")
