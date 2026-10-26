import pytest

from chunkguard.backend import Settings, chunker_from_env
from chunkguard.pic import PICChunker, _merge_short, _runs, _split_long

VOCABULARY = ["resort", "meer", "wohnung", "preis", "wetter", "parkplatz", "hund"]


def bag_of_words(texts: list[str]) -> list[list[float]]:
    return [[float(t.lower().count(word)) for word in VOCABULARY] for t in texts]


class Fake:
    """Counts calls so tests can assert one LLM call per document."""

    def __init__(self, summary: str) -> None:
        self.summary = summary
        self.summarize_calls = 0
        self.embed_calls: list[list[str]] = []

    def summarize(self, text: str, language: str) -> str:
        self.summarize_calls += 1
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
    return PICChunker(fake.summarize, fake.embed, **limits), fake


def test_boundaries_where_relevance_flips():
    chunker, fake = make()
    chunks = chunker.split("d1", DOCUMENT, "de")

    assert [c.relevant for c in chunks] == [True, False, True]
    assert [(c.sentence_start, c.sentence_end) for c in chunks] == [(0, 2), (2, 4), (4, 5)]
    assert chunks[1].text == "Gestern war das Wetter schlecht. Ein Hund lief über den Parkplatz."
    assert chunks[0].similarity > chunks[1].similarity
    assert fake.summarize_calls == 1
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
    assert fake.summarize_calls == 0 and fake.embed_calls == []


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


def test_from_env_builds_azure_and_compatible_clients(tmp_path, monkeypatch):
    for key in list(Settings.model_fields):
        monkeypatch.delenv(f"CHUNKGUARD_{key.upper()}", raising=False)
    env = tmp_path / ".env"
    env.write_text(
        "CHUNKGUARD_LLM_PROVIDER=azure\n"
        "CHUNKGUARD_LLM_MODEL=gpt-deploy\n"
        "CHUNKGUARD_LLM_API_KEY=k\n"
        "CHUNKGUARD_LLM_BASE_URL=https://x.openai.azure.com\n"
        "CHUNKGUARD_LLM_API_VERSION=2024-10-21\n"
        "CHUNKGUARD_EMBEDDING_MODEL=bge-m3\n"
        "CHUNKGUARD_EMBEDDING_API_KEY=k\n"
        "CHUNKGUARD_EMBEDDING_BASE_URL=http://localhost:8080/v1\n"
        "CHUNKGUARD_EMBEDDING_API_VERSION=\n"
        "CHUNKGUARD_MAX_CHUNK_CHARS=1500\n"
    )
    chunker = chunker_from_env(str(env))
    assert chunker.max_chars == 1500 and chunker.min_chars == 200

    llm = chunker.summarize.__closure__
    embedder = chunker.embed.__closure__
    clients = {type(c.cell_contents).__name__ for c in llm + embedder}
    assert {"AzureOpenAI", "OpenAI"} <= clients


def test_missing_model_config_fails(tmp_path, monkeypatch):
    for key in list(Settings.model_fields):
        monkeypatch.delenv(f"CHUNKGUARD_{key.upper()}", raising=False)
    env = tmp_path / ".env"
    env.write_text("CHUNKGUARD_LLM_API_KEY=k\n")
    with pytest.raises(ValueError):
        chunker_from_env(str(env))
