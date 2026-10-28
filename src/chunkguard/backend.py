"""Any LLM and embedding provider via LiteLLM, configured from environment variables.

Needs the extra: pip install "chunkguard[litellm]". Models are LiteLLM strings
such as "openai/gpt-4.1-mini", "anthropic/claude-sonnet-5-5", "gemini/gemini-2.5-flash",
"azure/<deployment>", "bedrock/...", "ollama/llama3.1" or "mistral/mistral-embed".

Settings come from the process environment; a .env file is read only when its path is
passed. LiteLLM itself loads a .env on import unless LITELLM_MODE is set to something
other than DEV, so this module sets LITELLM_MODE=PRODUCTION if it is unset.
"""

from __future__ import annotations

import asyncio
import os
from typing import Any, Literal

os.environ.setdefault("LITELLM_MODE", "PRODUCTION")

try:
    import litellm
    from pydantic import AliasChoices, Field, ValidationError
    from pydantic_settings import BaseSettings, SettingsConfigDict
except ImportError as error:
    raise ImportError(
        'chunker_from_env needs the litellm extra: pip install "chunkguard[litellm]"'
    ) from error

from chunkguard.pic import (
    AsyncComplete,
    AsyncEmbed,
    AsyncPICChunker,
    Complete,
    Embed,
    Length,
    PICChunker,
)
from chunkguard.tokens import token_length

# Most providers accept far more, but Cohere caps a request at 96 inputs.
DEFAULT_EMBED_BATCH_SIZE = 64
# Embedding requests in flight at once (async only); free tiers reject bursts.
DEFAULT_EMBED_CONCURRENCY = 4
# Keeps a request well below the common 300k-token limit, even for dense scripts.
EMBED_BATCH_CHARS = 400_000
# Only the embedding input is cut; the chunk text always keeps the whole sentence.
EMBED_INPUT_MAX_CHARS = 8_000


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="CHUNKGUARD_", env_file=None, env_ignore_empty=True, extra="ignore"
    )

    # Unset keys fall back to the provider's own variables (OPENAI_API_KEY, AWS_*, ...).
    llm_model: str
    llm_api_key: str | None = None
    llm_api_base: str | None = None
    llm_api_version: str | None = None

    embedding_model: str
    embedding_api_key: str | None = None
    embedding_api_base: str | None = None
    embedding_api_version: str | None = None
    embedding_batch_size: int = DEFAULT_EMBED_BATCH_SIZE
    embedding_concurrency: int = Field(DEFAULT_EMBED_CONCURRENCY, ge=1)  # async only

    # Per request. LiteLLM's own defaults are 600 s and no retries.
    request_timeout: float = 120
    max_retries: int = 2

    # Limits in LENGTH_UNIT. The *_CHARS names from 0.1 are still read.
    max_chunk_length: int = Field(  # 0 = no limit
        2000,
        validation_alias=AliasChoices("CHUNKGUARD_MAX_CHUNK_LENGTH", "CHUNKGUARD_MAX_CHUNK_CHARS"),
    )
    min_chunk_length: int = Field(  # e.g. 200 merges shorter chunks into a neighbour
        0,
        validation_alias=AliasChoices("CHUNKGUARD_MIN_CHUNK_LENGTH", "CHUNKGUARD_MIN_CHUNK_CHARS"),
    )
    length_unit: Literal["chars", "tokens"] = "chars"
    token_encoding: str = "o200k_base"  # tiktoken encoding, LENGTH_UNIT=tokens only
    summary_max_chars: int = 100_000  # 0 = send the whole document


def _connection(
    api_key: str | None, api_base: str | None, api_version: str | None, s: Settings
) -> dict:
    options = {"api_key": api_key, "api_base": api_base, "api_version": api_version}
    return {key: value for key, value in options.items() if value is not None} | {
        "timeout": s.request_timeout,
        "num_retries": s.max_retries,
    }


def _batches(texts: list[str], size: int) -> list[list[str]]:
    batches: list[list[str]] = []
    chars = 0
    for text in (t[:EMBED_INPUT_MAX_CHARS] for t in texts):
        if not batches or len(batches[-1]) == size or chars + len(text) > EMBED_BATCH_CHARS:
            batches.append([])
            chars = 0
        batches[-1].append(text)
        chars += len(text)
    return batches


def _vectors(response: Any) -> list[list[float]]:
    items = sorted(response.data, key=lambda item: item["index"])
    return [item["embedding"] for item in items]


def _messages(prompt: str) -> list[dict[str, str]]:
    return [{"role": "user", "content": prompt}]


def litellm_complete(model: str, **connection: Any) -> Complete:
    def complete(prompt: str) -> str:
        response = litellm.completion(model=model, messages=_messages(prompt), **connection)
        return response.choices[0].message.content or ""

    return complete


def litellm_embed(
    model: str, batch_size: int = DEFAULT_EMBED_BATCH_SIZE, **connection: Any
) -> Embed:
    def embed(texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for batch in _batches(texts, batch_size):
            vectors.extend(_vectors(litellm.embedding(model=model, input=batch, **connection)))
        return vectors

    return embed


def async_litellm_complete(model: str, **connection: Any) -> AsyncComplete:
    async def complete(prompt: str) -> str:
        response = await litellm.acompletion(model=model, messages=_messages(prompt), **connection)
        return response.choices[0].message.content or ""

    return complete


def async_litellm_embed(
    model: str,
    batch_size: int = DEFAULT_EMBED_BATCH_SIZE,
    concurrency: int = DEFAULT_EMBED_CONCURRENCY,
    **connection: Any,
) -> AsyncEmbed:
    """Send the batches in parallel, at most `concurrency` at a time, keeping their order."""
    if concurrency < 1:
        raise ValueError("concurrency must be at least 1")

    async def embed(texts: list[str]) -> list[list[float]]:
        slots = asyncio.Semaphore(concurrency)

        async def one(batch: list[str]) -> list[list[float]]:
            async with slots:
                return _vectors(await litellm.aembedding(model=model, input=batch, **connection))

        results = await asyncio.gather(*(one(b) for b in _batches(texts, batch_size)))
        return [vector for batch in results for vector in batch]

    return embed


def _parts(env_file: str | None) -> tuple[Settings, dict, dict, dict]:
    if env_file is not None and not os.path.isfile(env_file):
        raise FileNotFoundError(f"env file not found: {env_file}")
    try:
        s = Settings(_env_file=env_file)
    except ValidationError as error:
        names = ", ".join(f"CHUNKGUARD_{str(e['loc'][0]).upper()}" for e in error.errors())
        where = f"{env_file} or the environment" if env_file else "the environment"
        raise ValueError(f"invalid or missing settings: {names}; set them in {where}") from error
    llm = _connection(s.llm_api_key, s.llm_api_base, s.llm_api_version, s)
    embedder = _connection(s.embedding_api_key, s.embedding_api_base, s.embedding_api_version, s)
    length: Length = len if s.length_unit == "chars" else token_length(s.token_encoding)
    options = {
        "max_length": s.max_chunk_length or None,
        "min_length": s.min_chunk_length,
        "length": length,
        "summary_max_chars": s.summary_max_chars or None,
    }
    return s, llm, embedder, options


def chunker_from_env(env_file: str | None = None) -> PICChunker:
    """Build a PICChunker from CHUNKGUARD_* variables, plus env_file if given."""
    s, llm, embedder, options = _parts(env_file)
    return PICChunker(
        litellm_complete(s.llm_model, **llm),
        litellm_embed(s.embedding_model, s.embedding_batch_size, **embedder),
        **options,
    )


def async_chunker_from_env(env_file: str | None = None) -> AsyncPICChunker:
    """Build an AsyncPICChunker from CHUNKGUARD_* variables, plus env_file if given."""
    s, llm, embedder, options = _parts(env_file)
    return AsyncPICChunker(
        async_litellm_complete(s.llm_model, **llm),
        async_litellm_embed(
            s.embedding_model, s.embedding_batch_size, s.embedding_concurrency, **embedder
        ),
        **options,
    )
