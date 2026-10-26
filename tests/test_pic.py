import asyncio
import re

import pytest

from chunkguard.pic import (
    SUMMARY_PROMPT,
    AsyncPICChunker,
    PICChunker,
    _excerpt,
    _merge_short,
    _runs,
    _split_long,
)

VOCABULARY = ["resort", "meer", "wohnung", "preis", "wetter", "parkplatz", "hund"]


def bag_of_words(texts: list[str]) -> list[list[float]]:
    return [[float(t.lower().count(word)) for word in VOCABULARY] for t in texts]


class Fake:
    """Counts calls so tests can assert one LLM call per document."""

    def __init__(self, summary: str) -> None:
        self.summary = summary
        self.prompts: list[str] = []
        self.embed_calls: list[list[str]] = []

    def complete(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self.summary

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.embed_calls.append(texts)
        return bag_of_words(texts)


DOCUMENT = (
    "Das Resort liegt am Meer. "
    "Jede Wohnung im Resort hat Meerblick. "
    "Gestern war das Wetter schlecht. "
    "Ein Hund lief über den Parkplatz. "
    "Der Preis jeder Wohnung steht im Resort aus."
)


def make(summary: str = "Resort am Meer, Wohnung, Preis.", **limits) -> tuple[PICChunker, Fake]:
    fake = Fake(summary)
    limits = {"min_chars": 0, "max_chars": 10_000} | limits
    return PICChunker(fake.complete, fake.embed, **limits), fake


def test_boundaries_where_relevance_flips():
    chunker, fake = make()
    chunks = chunker.split("d1", DOCUMENT, "de")

    assert [c.relevant for c in chunks] == [True, False, True]
    assert [(c.sentence_start, c.sentence_end) for c in chunks] == [(0, 2), (2, 4), (4, 5)]
    assert chunks[1].text == "Gestern war das Wetter schlecht. Ein Hund lief über den Parkplatz."
    assert chunks[0].similarity > chunks[1].similarity
    assert fake.prompts == [SUMMARY_PROMPT.format(text=DOCUMENT, language="de")]
    assert fake.embed_calls[0][0] == fake.summary


def test_chunks_are_exact_slices_in_order_and_lose_nothing():
    chunker, _ = make()
    text = "  " + DOCUMENT.replace(". ", ".\n\n") + "\n"
    chunks = chunker.split("d1", text, "de")

    assert [c.index for c in chunks] == list(range(len(chunks)))
    for chunk in chunks:
        assert text[chunk.start : chunk.end] == chunk.text
        assert chunk.doc_id == "d1"
        assert chunk.summary == "Resort am Meer, Wohnung, Preis."
    gaps = [text[: chunks[0].start]] + [
        text[a.end : b.start] for a, b in zip(chunks, chunks[1:], strict=False)
    ]
    gaps.append(text[chunks[-1].end :])
    assert all(gap.strip() == "" for gap in gaps)


def test_empty_document_makes_no_calls():
    chunker, fake = make()
    assert chunker.split("d1", " \n ", "de") == []
    assert fake.prompts == [] and fake.embed_calls == []


def test_runs():
    assert _runs([True, True, False, True]) == [(0, 2), (2, 3), (3, 4)]
    assert _runs([False]) == [(0, 1)]


def spans_of(lengths: list[int]) -> list[tuple[int, int]]:
    """Sentence spans of the given lengths, separated by one space."""
    spans, pos = [], 0
    for n in lengths:
        spans.append((pos, pos + n))
        pos += n + 1
    return spans


def test_split_long_into_near_equal_pieces_at_sentence_boundaries():
    spans = spans_of([30] * 10)  # 309 chars
    pieces = _split_long((0, 10), spans, max_chars=200)
    assert pieces == [(0, 5), (5, 10)]


def test_single_overlong_sentence_stays_whole():
    spans = spans_of([50, 500, 50])
    pieces = _split_long((0, 3), spans, max_chars=100)
    assert (1, 2) in pieces
    assert pieces[0][0] == 0 and pieces[-1][1] == 3


def test_merge_short_into_neighbour_with_closest_similarity():
    spans = spans_of([100, 10, 100])
    similarities = [0.9, 0.2, 0.3]
    groups = _merge_short([(0, 1), (1, 2), (2, 3)], spans, similarities, 50, 1000)
    assert groups == [(0, 1), (1, 3)]


def test_merge_short_never_exceeds_max():
    spans = spans_of([100, 10, 100])
    groups = _merge_short([(0, 1), (1, 2), (2, 3)], spans, [0.9, 0.2, 0.3], 50, 105)
    assert groups == [(0, 1), (1, 2), (2, 3)]


def test_merged_chunk_flag_follows_its_mean_similarity():
    # Short irrelevant sentence between two long relevant runs gets absorbed.
    text = "Resort Meer Wohnung Preis. " * 4 + "Hund. " + "Resort Meer Wohnung Preis. " * 4
    chunker, _ = make(min_chars=20)
    chunks = chunker.split("d1", text.strip(), "de")
    assert len(chunks) == 2
    assert all(c.relevant for c in chunks)


def test_paper_mode_without_length_limits():
    chunker, _ = make(max_chars=None, min_chars=0)
    chunks = chunker.split("d1", DOCUMENT, "de")
    assert [(c.sentence_start, c.sentence_end) for c in chunks] == [(0, 2), (2, 4), (4, 5)]


def test_custom_sentence_splitter():
    fake = Fake("Resort")
    lines = lambda text, language: [  # noqa: E731
        (m.start(), m.end()) for m in re.finditer(r"[^|]+", text)
    ]
    chunker = PICChunker(fake.complete, fake.embed, min_chars=0, sentence_splitter=lines)
    chunks = chunker.split("d1", "Resort A|Hund B|Resort C", "xx")
    assert [c.text for c in chunks] == ["Resort A", "Hund B", "Resort C"]


def test_long_document_summary_sees_evenly_spread_excerpt():
    text = "".join(f"Satz {i:05d} ist da. " for i in range(10_000))
    seen = []
    chunker = PICChunker(
        lambda prompt: seen.append(prompt) or "Satz", bag_of_words, summary_max_chars=8_000
    )
    chunker.split("d1", text, "de")
    excerpt = seen[0]
    assert len(excerpt) <= 8_000 + 7 * len("\n[…]\n") + len(SUMMARY_PROMPT)
    assert "Satz 00000" in excerpt and "Satz 09999" in excerpt and "Satz 05" in excerpt
    assert _excerpt("kurz", 8_000) == "kurz"


def test_empty_summary_is_an_error():
    chunker = PICChunker(lambda prompt: "  ", bag_of_words)
    with pytest.raises(ValueError, match="empty summary"):
        chunker.split("d1", DOCUMENT, "de")


def test_wrong_number_of_embeddings_is_an_error():
    chunker = PICChunker(lambda prompt: "Resort", lambda texts: [[1.0]])
    with pytest.raises(ValueError, match="embeddings"):
        chunker.split("d1", DOCUMENT, "de")


@pytest.mark.parametrize(
    "options",
    [{"max_chars": 0}, {"min_chars": -1}, {"min_chars": 300, "max_chars": 200}],
)
def test_invalid_limits(options):
    with pytest.raises(ValueError):
        PICChunker(lambda prompt: "", bag_of_words, **options)


def test_async_chunker_matches_sync():
    fake = Fake("Resort am Meer, Wohnung, Preis.")

    async def complete(prompt):
        return fake.complete(prompt)

    async def embed(texts):
        return fake.embed(texts)

    sync_chunks, _ = make()
    expected = sync_chunks.split("d1", DOCUMENT, "de")
    chunker = AsyncPICChunker(complete, embed, min_chars=0, max_chars=10_000)
    assert asyncio.run(chunker.split("d1", DOCUMENT, "de")) == expected
    assert len(fake.prompts) == 1


def test_async_summary_sees_excerpt():
    seen = []

    async def complete(prompt):
        seen.append(prompt)
        return "Satz"

    async def embed(texts):
        return bag_of_words(texts)

    chunker = AsyncPICChunker(complete, embed, summary_max_chars=1_000)
    asyncio.run(chunker.split("d1", "Ein Satz ist da. " * 1_000, "de"))
    assert len(seen[0]) < 1_100 + len(SUMMARY_PROMPT)


def test_custom_summary_prompt():
    fake = Fake("Resort")
    chunker = PICChunker(fake.complete, fake.embed, summary_prompt="[{language}] {text}")
    chunker.split("d1", "Resort am Meer.", "it")
    assert fake.prompts == ["[it] Resort am Meer."]
