"""PIC: Pseudo-Instruction for document Chunking (Wang et al., Findings of ACL 2025).

An LLM summary of the document stands in for the unknown user queries. Every
sentence is compared with that summary; the mean similarity is the threshold.
Runs of neighbouring sentences on the same side of the threshold form a chunk,
so boundaries sit exactly where the text turns towards or away from the topic.

Splitting long chunks (max_length) and merging short ones (min_length) are not
part of the paper. Merging is off by default; max_length=None switches splitting
off. Lengths are measured by ``length``, characters unless you pass a token counter.
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Awaitable, Callable
from statistics import fmean

from pydantic import BaseModel

from chunkguard.sentences import split_sentences

Span = tuple[int, int]
# (text, language) -> sentence spans, see chunkguard.sentences
SentenceSplitter = Callable[[str, str], list[Span]]
# Any LLM: prompt -> completion text
Complete = Callable[[str], str]
AsyncComplete = Callable[[str], Awaitable[str]]
# Any embedding model: texts -> one vector per text, same order
Embed = Callable[[list[str]], list[list[float]]]
AsyncEmbed = Callable[[list[str]], Awaitable[list[list[float]]]]
# text -> its length in whatever unit the limits use (len for characters)
Length = Callable[[str], int]

# Summarizer prompt from the PIC paper (Appendix C), plus the output language.
SUMMARY_PROMPT = (
    "Read this passage:\n{text}\n"
    "Write a summary for this passage. Your summary should be concise and informative. "
    "Give the summary directly, without any other text. "
    "Write it in the language of the passage (language code: {language})."
)

# A sentence range [first, end) within the document.
Group = tuple[int, int]

# Pieces of a long document the summarizer sees, spread evenly over the text.
_EXCERPT_PIECES = 8


class Chunk(BaseModel):
    doc_id: str
    language: str  # as passed to split()
    index: int
    text: str
    start: int  # character offset in the input text, inclusive
    end: int  # character offset in the input text, exclusive
    sentence_start: int  # inclusive
    sentence_end: int  # exclusive
    # Sentences at or above the document mean make up at least half the text.
    # Before merging every chunk is uniform; a merge may absorb a short opposite run.
    relevant: bool
    similarity: float  # mean cosine similarity of the chunk's sentences to the summary
    summary: str


class _Chunker:
    def __init__(
        self,
        *,
        max_length: int | None = 2000,
        min_length: int = 0,
        length: Length = len,
        summary_max_chars: int | None = 100_000,
        summary_prompt: str = SUMMARY_PROMPT,
        sentence_splitter: SentenceSplitter = split_sentences,
        max_chars: int | None = None,
        min_chars: int | None = None,
    ) -> None:
        if max_chars is not None or min_chars is not None:
            warnings.warn(
                "max_chars/min_chars were renamed to max_length/min_length in 0.2.0",
                DeprecationWarning,
                stacklevel=3,
            )
            max_length = max_chars if max_chars is not None else max_length
            min_length = min_chars if min_chars is not None else min_length
        if max_length is not None and max_length < 1:
            raise ValueError("max_length must be positive or None")
        if min_length < 0 or (max_length is not None and min_length > max_length):
            raise ValueError("min_length must be between 0 and max_length")
        if summary_max_chars is not None and summary_max_chars < _EXCERPT_PIECES:
            raise ValueError(f"summary_max_chars must be at least {_EXCERPT_PIECES} or None")
        self.max_length = max_length
        self.min_length = min_length
        self.length = length
        self.summary_max_chars = summary_max_chars
        self.summary_prompt = summary_prompt
        self.sentence_splitter = sentence_splitter

    def _prompt(self, text: str, language: str) -> str:
        excerpt = _excerpt(text, self.summary_max_chars)
        return self.summary_prompt.format(text=excerpt, language=language)

    def _chunks(
        self,
        doc_id: str,
        text: str,
        language: str,
        spans: list[Span],
        summary: str,
        vectors: list[list[float]],
    ) -> list[Chunk]:
        if not summary:
            raise ValueError(f"LLM returned an empty summary for {doc_id!r}")
        if len(vectors) != len(spans) + 1:
            raise ValueError(f"expected {len(spans) + 1} embeddings, got {len(vectors)}")

        reference, *sentence_vectors = vectors
        similarities = [_cosine(reference, vector) for vector in sentence_vectors]
        threshold = fmean(similarities)
        sizes = [self.length(text[s:e]) for s, e in spans]
        max_length = math.inf if self.max_length is None else self.max_length

        groups = _runs([s >= threshold for s in similarities])
        groups = [p for g in groups for p in _split_long(g, sizes, max_length)]
        groups = _merge_short(groups, sizes, similarities, self.min_length, max_length)

        chunks = []
        for index, (first, end) in enumerate(groups):
            start, stop = spans[first][0], spans[end - 1][1]
            relevant_size = sum(
                n
                for n, sim in zip(sizes[first:end], similarities[first:end], strict=True)
                if sim >= threshold
            )
            chunks.append(
                Chunk(
                    doc_id=doc_id,
                    language=language,
                    index=index,
                    text=text[start:stop],
                    start=start,
                    end=stop,
                    sentence_start=first,
                    sentence_end=end,
                    relevant=2 * relevant_size >= sum(sizes[first:end]),
                    similarity=fmean(similarities[first:end]),
                    summary=summary,
                )
            )
        return chunks


class PICChunker(_Chunker):
    """Text in, chunks out. Swappable with any splitter of the same signature.

    Costs one LLM call and one embed call (for all sentences) per document.
    Pass ``summary=`` to reuse a stored summary and skip the LLM call.
    """

    def __init__(self, complete: Complete, embed: Embed, **options) -> None:
        super().__init__(**options)
        self.complete = complete
        self.embed = embed

    def split(
        self, doc_id: str, text: str, language: str, *, summary: str | None = None
    ) -> list[Chunk]:
        spans = self.sentence_splitter(text, language)
        if not spans:
            return []
        if summary is None:
            summary = self.complete(self._prompt(text, language))
        summary = (summary or "").strip()
        vectors = self.embed([summary, *(text[s:e] for s, e in spans)])
        return self._chunks(doc_id, text, language, spans, summary, vectors)


class AsyncPICChunker(_Chunker):
    """PICChunker with async complete/embed for asyncio indexing pipelines."""

    def __init__(self, complete: AsyncComplete, embed: AsyncEmbed, **options) -> None:
        super().__init__(**options)
        self.complete = complete
        self.embed = embed

    async def split(
        self, doc_id: str, text: str, language: str, *, summary: str | None = None
    ) -> list[Chunk]:
        spans = self.sentence_splitter(text, language)
        if not spans:
            return []
        if summary is None:
            summary = await self.complete(self._prompt(text, language))
        summary = (summary or "").strip()
        vectors = await self.embed([summary, *(text[s:e] for s, e in spans)])
        return self._chunks(doc_id, text, language, spans, summary, vectors)


def _excerpt(text: str, limit: int | None) -> str:
    """Shorten text to about limit characters, taking evenly spread pieces.

    Keeps the summary within the LLM's context window while every part of the
    document still contributes to it.
    """
    if limit is None or len(text) <= limit:
        return text
    size = limit // _EXCERPT_PIECES
    step = (len(text) - size) / (_EXCERPT_PIECES - 1)
    return "\n[…]\n".join(
        text[round(i * step) : round(i * step) + size] for i in range(_EXCERPT_PIECES)
    )


def _cosine(a: list[float], b: list[float]) -> float:
    norm = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
    return sum(x * y for x, y in zip(a, b, strict=True)) / norm if norm else 0.0


def _runs(flags: list[bool]) -> list[Group]:
    """Split the sentence sequence wherever the flag flips."""
    groups, first = [], 0
    for i in range(1, len(flags)):
        if flags[i] != flags[i - 1]:
            groups.append((first, i))
            first = i
    groups.append((first, len(flags)))
    return groups


def _size(sizes: list[int], group: Group) -> int:
    """Length of a group: the sum of its sentences (whitespace between them not counted)."""
    first, end = group
    return sum(sizes[first:end])


def _split_long(group: Group, sizes: list[int], max_length: float) -> list[Group]:
    """Split at sentence boundaries into near-equal pieces of at most max_length.

    A single sentence longer than max_length stays whole.
    """
    total = _size(sizes, group)
    if total <= max_length:
        return [group]
    target = total / math.ceil(total / max_length)
    first, end = group
    pieces, start = [], first
    for i in range(first + 1, end):
        without, with_next = _size(sizes, (start, i)), _size(sizes, (start, i + 1))
        if with_next > max_length or with_next - target > target - without:
            pieces.append((start, i))
            start = i
    pieces.append((start, end))
    return pieces


def _merge_short(
    groups: list[Group],
    sizes: list[int],
    similarities: list[float],
    min_length: int,
    max_length: float,
) -> list[Group]:
    """Merge chunks below min_length into the neighbour with the closest mean similarity.

    max_length wins over min_length: a merge that would exceed it is skipped.
    """
    groups = list(groups)

    def score(group: Group) -> float:
        return fmean(similarities[group[0] : group[1]])

    while True:
        for k in sorted(range(len(groups)), key=lambda k: _size(sizes, groups[k])):
            if _size(sizes, groups[k]) >= min_length:
                return groups
            # Merging with neighbour k-1 or k+1 means merging the pair starting at lo.
            pairs = [
                lo
                for lo in (k - 1, k)
                if 0 <= lo < len(groups) - 1
                and _size(sizes, (groups[lo][0], groups[lo + 1][1])) <= max_length
            ]
            if pairs:
                lo = min(
                    pairs, key=lambda lo: abs(score(groups[2 * lo + 1 - k]) - score(groups[k]))
                )
                groups[lo : lo + 2] = [(groups[lo][0], groups[lo + 1][1])]
                break
        else:
            return groups
