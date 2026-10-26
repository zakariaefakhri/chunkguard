"""Rule-based sentence splitter that returns character spans, so no text is lost."""

import re

# Lowercase tokens that end with a period without ending a sentence.
# Single letters ("z. B.", "J. R. R.", "e.g.") are handled generically.
_ABBREVIATIONS = {
    "de": set(
        "abs art bspw bzw ca dr evtl ff fr gem ggf hr inkl jh lt max min mind mio mrd nr prof "
        "st str tel tsd usw vgl zzgl".split()
    ),
    "en": set("approx cf dr etc fig inc jr ltd mr mrs ms no prof sr st vs".split()),
    "it": set("arch art avv dott ecc es ing pag prof sig sigg tel".split()),
    "hu": set("bt dr ill kb kft nyrt pl stb zrt".split()),
}

# Languages that write ordinals as "3. Oktober" / "2. emelet".
_ORDINAL_LANGUAGES = {"de", "hu"}

# word before the terminator, terminator plus closing quotes, first char after the gap
_SENTENCE_END = re.compile(
    r"(?P<word>[^\s.!?…]*)(?P<end>[.!?…]+[\"'»«“”’)\]]*)(?=\s+(?P<next>\S)|\s*$)"
)


def split_sentences(text: str, language: str) -> list[tuple[int, int]]:
    """Return the (start, end) character spans of the sentences in ``text``.

    Line breaks always end a sentence, so headings and list items stand alone.
    Spans exclude surrounding whitespace; ``text[start:end]`` is the sentence.
    """
    lang = language.split("-")[0].lower()
    abbreviations = _ABBREVIATIONS.get(lang, set())
    ordinals = lang in _ORDINAL_LANGUAGES

    cuts = {i for i, char in enumerate(text) if char == "\n"}
    for match in _SENTENCE_END.finditer(text):
        if _is_boundary(match, abbreviations, ordinals):
            cuts.add(match.end())

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


def _is_boundary(match: re.Match[str], abbreviations: set[str], ordinals: bool) -> bool:
    following = match.group("next")
    if following is None or following.islower():
        return False
    if any(char in "!?…" for char in match.group("end")):
        return True
    word = match.group("word").lstrip("([\"'»«„“")
    if len(word) == 1 and word.isalpha():
        return False
    if word.lower() in abbreviations:
        return False
    if ordinals and word.isdigit() and len(word) <= 3:
        return False
    return True
