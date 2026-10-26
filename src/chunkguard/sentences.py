"""Sentence splitting that returns character spans, so no text is lost.

Boundaries come from sentencex (Wikimedia), which carries abbreviation and
punctuation rules for many languages and falls back to related languages for
the rest. On top of that, every line break ends a sentence, so headings and
list items stand alone.
"""

import sentencex


def split_sentences(text: str, language: str) -> list[tuple[int, int]]:
    """Return the (start, end) character spans of the sentences in ``text``.

    ``language`` is a BCP 47 code such as "en" or "pt-BR"; only the primary
    subtag is used. Spans exclude surrounding whitespace, so ``text[start:end]``
    is the sentence, and whitespace-only pieces are dropped.
    """
    primary = language.split("-")[0].lower()
    cuts = {i + 1 for i, char in enumerate(text) if char == "\n"}
    for boundary in sentencex.get_sentence_boundaries(primary, text):
        if not _continues_lowercase(text, boundary["end_index"]):
            cuts.add(boundary["end_index"])

    spans: list[tuple[int, int]] = []
    start = 0
    for cut in [*sorted(cuts), len(text)]:
        piece = text[start:cut]
        left = len(piece) - len(piece.lstrip())
        right = len(piece.rstrip())
        if right > left:
            spans.append((start + left, start + right))
        start = cut
    return spans


def _continues_lowercase(text: str, position: int) -> bool:
    """True if the same line goes on with a lowercase letter, i.e. after an abbreviation.

    sentencex misses this for some abbreviations and for non-ASCII letters such
    as "ü". Merging two sentences costs less than cutting one in half.
    """
    rest = text[position:].lstrip(" \t")
    return bool(rest) and rest[0].islower()
