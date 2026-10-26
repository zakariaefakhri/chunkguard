"""PIC: Pseudo-Instruction for document Chunking (Wang et al., ACL 2025).

An LLM summary of the document stands in for the unknown user queries. Every
sentence is compared with that summary; the mean similarity is the threshold.
Runs of neighbouring sentences on the same side of the threshold form a chunk,
so boundaries sit exactly where the text turns towards or away from the topic.
"""

import math
from collections.abc import Callable
from statistics import fmean

from pydantic import BaseModel

from chunkguard.sentences import split_sentences

# (document text, language) -> summary
Summarize = Callable[[str, str], str]
# texts -> one vector per text, same order
Embed = Callable[[list[str]], list[list[float]]]

# A sentence range [first, end) within the document.
Group = tuple[int, int]


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


class PICChunker:
    """Text in, chunks out. Swappable with any splitter of the same signature."""

    def __init__(
        self,
        summarize: Summarize,
        embed: Embed,
        max_chars: int = 2000,
        min_chars: int = 200,
    ) -> None:
        self.summarize = summarize
        self.embed = embed
        self.max_chars = max_chars
        self.min_chars = min_chars

    def split(self, doc_id: str, text: str, language: str) -> list[Chunk]:
        spans = split_sentences(text, language)
        if not spans:
            return []

        summary = self.summarize(text, language)
        reference, *vectors = self.embed([summary, *(text[s:e] for s, e in spans)])
        similarities = [_cosine(reference, vector) for vector in vectors]
        threshold = fmean(similarities)

        groups = _runs([s >= threshold for s in similarities])
        groups = [p for g in groups for p in _split_long(g, spans, self.max_chars)]
        groups = _merge_short(groups, spans, similarities, self.min_chars, self.max_chars)

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


def _length(spans: list[tuple[int, int]], group: Group) -> int:
    first, end = group
    return spans[end - 1][1] - spans[first][0]


def _split_long(group: Group, spans: list[tuple[int, int]], max_chars: int) -> list[Group]:
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
    spans: list[tuple[int, int]],
    similarities: list[float],
    min_chars: int,
    max_chars: int,
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
