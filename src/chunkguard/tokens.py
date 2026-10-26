"""Token-based lengths for the chunk limits. Needs: pip install "chunkguard[tokens]"."""

from __future__ import annotations

from chunkguard.pic import Length


def token_length(encoding: str = "o200k_base") -> Length:
    """Return a length function that counts tiktoken tokens.

    ``o200k_base`` is the encoding of current OpenAI models; other providers'
    tokenizers differ by some percent, which the limits tolerate.
    """
    try:
        import tiktoken
    except ImportError as error:
        raise ImportError(
            'token lengths need the tokens extra: pip install "chunkguard[tokens]"'
        ) from error
    encoder = tiktoken.get_encoding(encoding)
    return lambda text: len(encoder.encode(text))
