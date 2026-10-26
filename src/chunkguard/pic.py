"""PIC: Pseudo-Instruction for document Chunking (Wang et al., Findings of ACL 2025).

An LLM summary of the document stands in for the unknown user queries. Every
sentence is compared with that summary; the mean similarity is the threshold.
Runs of neighbouring sentences on the same side of the threshold form a chunk,
so boundaries sit exactly where the text turns towards or away from the topic.

Splitting long chunks (max_chars) and merging short ones (min_chars) are not
part of the paper. Merging is off by default; max_chars=None switches splitting off.
"""

from __future__ import annotations

import math
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
        max_chars: int | None = 2000,
        min_chars: int = 0,
        summary_max_chars: int | None = 100_000,
        summary_prompt: str = SUMMARY_PROMPT,
        sentence_splitter: SentenceSplitter = split_sentences,
    ) -> None:
        if max_chars is not None and max_chars < 1:
            raise ValueError("max_chars must be positive or None")
        if min_chars < 0 or (max_chars is not None and min_chars > max_chars):
            raise ValueError("min_chars must be between 0 and max_chars")
        if summary_max_chars is not None and summary_max_chars < _EXCERPT_PIECES:
            raise ValueError(f"summary_max_chars must be at least {_EXCERPT_PIECES} or None")
        self.max_chars = max_chars
        self.min_chars = min_chars
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
        max_chars = math.inf if self.max_chars is None else self.max_chars

        groups = _runs([s >= threshold for s in similarities])
        groups = [p for g in groups for p in _split_long(g, spans, max_chars)]
        groups = _merge_short(groups, spans, similarities, self.min_chars, max_chars)

        chunks = []
        for index, (first, end) in enumerate(groups):
            start, stop = spans[first][0], spans[end - 1][1]
            sizes = [e - s for s, e in spans[first:end]]
            relevant_chars = sum(
                n for n, sim in zip(sizes, similarities[first:end], strict=True) if sim >= threshold
            )
            chunks.append(
                Chunk(
                    doc_id=doc_id,
                    index=index,
                    text=text[start:stop],
                    start=start,
                    end=stop,
                    sentence_start=first,
                    sentence_end=end,
                    relevant=2 * relevant_chars >= sum(sizes),
                    similarity=fmean(similarities[first:end]),
                    summary=summary,
                )
            )
        return chunks


class PICChunker(_Chunker):
    """Text in, chunks out. Swappable with any splitter of the same signature.

    Costs one LLM call and one embed call (for all sentences) per document.
    """

    def __init__(self, complete: Complete, embed: Embed, **options) -> None:
        super().__init__(**options)
        self.complete = complete
        self.embed = embed

    def split(self, doc_id: str, text: str, language: str) -> list[Chunk]:
        spans = self.sentence_splitter(text, language)
        if not spans:
            return []
        summary = (self.complete(self._prompt(text, language)) or "").strip()
        vectors = self.embed([summary, *(text[s:e] for s, e in spans)])
        return self._chunks(doc_id, text, spans, summary, vectors)


class AsyncPICChunker(_Chunker):
    """PICChunker with async complete/embed for asyncio indexing pipelines."""

    def __init__(self, complete: AsyncComplete, embed: AsyncEmbed, **options) -> None:
        super().__init__(**options)
        self.complete = complete
        self.embed = embed

    async def split(self, doc_id: str, text: str, language: str) -> list[Chunk]:
        spans = self.sentence_splitter(text, language)
        if not spans:
            return []
        summary = (await self.complete(self._prompt(text, language)) or "").strip()
        vectors = await self.embed([summary, *(text[s:e] for s, e in spans)])
        return self._chunks(doc_id, text, spans, summary, vectors)


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


def _length(spans: list[Span], group: Group) -> int:
    first, end = group
    return spans[end - 1][1] - spans[first][0]


def _split_long(group: Group, spans: list[Span], max_chars: float) -> list[Group]:
    """Split at sentence boundaries into near-equal pieces of at most max_chars.

    A single sentence longer than max_chars stays whole.
    """
    total = _length(spans, group)
    if total <= max_chars:
        return [group]
    target = total / math.ceil(total / max_chars)
    first, end = group
    pieces, start = [], first
    for i in range(first + 1, end):
        without, with_next = _length(spans, (start, i)), _length(spans, (start, i + 1))
        if with_next > max_chars or with_next - target > target - without:
            pieces.append((start, i))
            start = i
    pieces.append((start, end))
    return pieces


def _merge_short(
    groups: list[Group],
    spans: list[Span],
    similarities: list[float],
    min_chars: int,
    max_chars: float,
) -> list[Group]:
    """Merge chunks below min_chars into the neighbour with the closest mean similarity.

    max_chars wins over min_chars: a merge that would exceed it is skipped.
    """
    groups = list(groups)

    def score(group: Group) -> float:
        return fmean(similarities[group[0] : group[1]])

    while True:
        for k in sorted(range(len(groups)), key=lambda k: _length(spans, groups[k])):
            if _length(spans, groups[k]) >= min_chars:
                return groups
            # Merging with neighbour k-1 or k+1 means merging the pair starting at lo.
            pairs = [
                lo
                for lo in (k - 1, k)
                if 0 <= lo < len(groups) - 1
                and _length(spans, (groups[lo][0], groups[lo + 1][1])) <= max_chars
            ]
            if pairs:
                lo = min(
                    pairs, key=lambda lo: abs(score(groups[2 * lo + 1 - k]) - score(groups[k]))
                )
                groups[lo : lo + 2] = [(groups[lo][0], groups[lo + 1][1])]
                break
        else:
            return groups
