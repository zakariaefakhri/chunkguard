"""OpenAI, Azure OpenAI or any OpenAI-compatible endpoint, configured from .env."""

from typing import Literal

from openai import AzureOpenAI, OpenAI
from pydantic_settings import BaseSettings, SettingsConfigDict

from chunkguard.pic import Embed, PICChunker, Summarize

EMBED_BATCH_SIZE = 256

SUMMARY_PROMPT = (
    "Summarize the following document in a few sentences. Cover its main topics "
    "and key facts, nothing else. Write in the language of the document "
    "(language code: {language}).\n\n{text}"
)

Provider = Literal["openai", "azure"]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="CHUNKGUARD_", env_file=".env", env_ignore_empty=True, extra="ignore"
    )

    llm_provider: Provider = "openai"
    llm_model: str  # Azure: deployment name
    llm_api_key: str
    llm_base_url: str | None = None  # compatible endpoint, or the Azure endpoint
    llm_api_version: str | None = None  # Azure only

    embedding_provider: Provider = "openai"
    embedding_model: str
    embedding_api_key: str
    embedding_base_url: str | None = None
    embedding_api_version: str | None = None

    max_chunk_chars: int = 2000
    min_chunk_chars: int = 200


def _client(
    provider: Provider, api_key: str, base_url: str | None, api_version: str | None
) -> OpenAI:
    if provider == "azure":
        return AzureOpenAI(azure_endpoint=base_url, api_key=api_key, api_version=api_version)
    return OpenAI(api_key=api_key, base_url=base_url)


def openai_summarize(client: OpenAI, model: str) -> Summarize:
    def summarize(text: str, language: str) -> str:
        prompt = SUMMARY_PROMPT.format(language=language, text=text)
        response = client.chat.completions.create(
            model=model, messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content or ""

    return summarize


def openai_embed(client: OpenAI, model: str) -> Embed:
    def embed(texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for i in range(0, len(texts), EMBED_BATCH_SIZE):
            response = client.embeddings.create(model=model, input=texts[i : i + EMBED_BATCH_SIZE])
            vectors.extend(item.embedding for item in sorted(response.data, key=lambda d: d.index))
        return vectors

    return embed


def chunker_from_env(env_file: str = ".env") -> PICChunker:
    s = Settings(_env_file=env_file)
    llm = _client(s.llm_provider, s.llm_api_key, s.llm_base_url, s.llm_api_version)
    embedder = _client(
        s.embedding_provider, s.embedding_api_key, s.embedding_base_url, s.embedding_api_version
    )
    return PICChunker(
        summarize=openai_summarize(llm, s.llm_model),
        embed=openai_embed(embedder, s.embedding_model),
        max_chars=s.max_chunk_chars,
        min_chars=s.min_chunk_chars,
    )
