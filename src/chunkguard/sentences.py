"""Sentence splitting that returns character spans, so no text is lost.

Boundaries come from sentencex (Wikimedia), which carries abbreviation and
punctuation rules for many languages and falls back to related languages for
the rest. chunkguard drops some of its boundaries again:

- after an abbreviation sentencex does not know yet (measured gaps, see below);
- before a lowercase letter, which never starts a sentence;
- after an ordinal number ("im 18. Jahrhundert"), but not after a house or phone
  number ("Hauptstr. 5.", "Tel. 069 123.").

Every line break ends a sentence, so headings and list items stand alone.
"""

import re
from functools import lru_cache

import sentencex

# Abbreviations sentencex 1.0.32 splits after although a name or noun follows,
# measured for German and English. Only those that never end a sentence: month,
# weekday and unit abbreviations ("am 5. Okt.", "zzgl. USt.") often do, so they
# stay sentence ends. Case-sensitive. To be reported upstream.
_EXTRA_ABBREVIATIONS = {
    "de": frozenset(
        "Abs. Abt. Anm. Aufl. Bd. Bhf. bspw. Dipl. dt. ehem. eigtl. einschl. entspr. Fa. Fam. "
        "geb. Gebr. gem. Hbf. Hrsg. inkl. Kap. Kfm. Kto. max. Pkt. röm. Tel. Vors. zw. "
        "zzgl.".split()
    ),
    "en": frozenset("Ms. approx. Tel. Vol.".split()),
}
# Street names: Hauptstr., Bahnhofstr.
_ABBREVIATION_SUFFIXES = {"de": ("str.",)}

_TOKEN = re.compile(r"\S+")
_SHORT_NUMBER = re.compile(r"\d{1,3}\.")
# Characters looked at around a boundary; slicing the whole text per boundary is quadratic.
_WINDOW = 80


def split_sentences(text: str, language: str) -> list[tuple[int, int]]:
    """Return the (start, end) character spans of the sentences in ``text``.

    ``language`` is a BCP 47 code such as "en" or "pt-BR"; only the primary
    subtag is used. Spans exclude surrounding whitespace, so ``text[start:end]``
    is the sentence, and whitespace-only pieces are dropped.
    """
    primary = language.split("-")[0].lower()
    cuts = {i + 1 for i, char in enumerate(text) if char == "\n"}
    boundaries = sorted(b["end_index"] for b in sentencex.get_sentence_boundaries(primary, text))
    for end in boundaries:
        if not (
            _continues_lowercase(text, end)
            or _ends_with_abbreviation(text, end, primary)
            or _ends_with_ordinal(text, end, primary, cuts)
        ):
            cuts.add(end)

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
    rest = text[position : position + _WINDOW].lstrip(" \t")
    return bool(rest) and rest[0].islower()


def _ends_with_abbreviation(text: str, position: int, language: str) -> bool:
    tokens = _TOKEN.findall(text, max(0, position - _WINDOW), position)
    return bool(tokens) and _is_extra_abbreviation(tokens[-1], language)


def _is_extra_abbreviation(word: str, language: str) -> bool:
    suffixes = _ABBREVIATION_SUFFIXES.get(language, ())
    return word in _EXTRA_ABBREVIATIONS.get(language, ()) or any(
        word.endswith(s) and len(word) > len(s) for s in suffixes
    )


@lru_cache(maxsize=4096)
def _ends_sentence(word: str, language: str) -> bool:
    """True if ``word`` (ending in a period) closes a sentence rather than abbreviating."""
    if _is_extra_abbreviation(word, language):
        return False
    return len(sentencex.get_sentence_boundaries(language, f"{word} Wort.")) > 1


def _ends_with_ordinal(text: str, position: int, language: str, cuts: set[int]) -> bool:
    """True if the piece ends in an ordinal such as "18.", not in a sentence-final number.

    A number of up to three digits plus a period is an ordinal in German,
    Hungarian, Danish, Czech and others; sentencex only knows the case before
    month names. It counts as an ordinal when it opens a sentence ("3. Mai",
    also right after a sentence sentencex did not end: "Er wurde Zweiter. 3.
    Platz"), follows a lowercase word ("im 18. Jahrhundert") or a capitalised
    word that opens the sentence ("Am 3. Mai", "Der 3. Platz"). After a digit
    group, an abbreviation or a noun inside the sentence it is a house, phone or
    page number, and the sentence ends ("Tel. 069 123.", "Hauptstr. 5.", "Nr. 5.",
    "Seite 5."). English "We sold 3. Then ..." is merged, a rare case.
    """
    window_start = max(0, position - _WINDOW)
    tokens = list(_TOKEN.finditer(text, window_start, position))
    if not tokens or not _SHORT_NUMBER.fullmatch(tokens[-1].group()):
        return False
    number = tokens[-1]
    if _opens_sentence(text, number.start(), cuts):
        return True
    if len(tokens) < 2:
        return False
    previous = tokens[-2]
    word = previous.group()
    if word.endswith(".") and word[:-1].isalpha() and _ends_sentence(word, language):
        return True
    return word.isalpha() and (word.islower() or _opens_sentence(text, previous.start(), cuts))


def _opens_sentence(text: str, position: int, cuts: set[int]) -> bool:
    """True if a sentence starts at position: text start, line break or an earlier cut."""
    before = position
    while before > 0 and text[before - 1] in " \t":
        before -= 1
    return before == 0 or text[before - 1] == "\n" or position in cuts or before in cuts
