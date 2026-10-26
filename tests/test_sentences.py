from chunkguard.sentences import split_sentences


def sentences(text: str, language: str = "de") -> list[str]:
    return [text[s:e] for s, e in split_sentences(text, language)]


def test_splits_on_terminal_punctuation():
    assert sentences("Das Haus ist neu. Es hat drei Zimmer! Wann? Morgen.") == [
        "Das Haus ist neu.",
        "Es hat drei Zimmer!",
        "Wann?",
        "Morgen.",
    ]


def test_keeps_abbreviations_initials_and_dates_together():
    text = "Die Wohnung Nr. 5 liegt z. B. am 3. Oktober frei, Stand 26.10.2026. Danach nicht."
    assert sentences(text) == [
        "Die Wohnung Nr. 5 liegt z. B. am 3. Oktober frei, Stand 26.10.2026.",
        "Danach nicht.",
    ]


def test_ordinals_only_in_ordinal_languages():
    assert len(sentences("We sold 3. Then we stopped.", "en")) == 2
    assert len(sentences("Am 3. Oktober ging es los.", "de")) == 1


def test_line_breaks_end_sentences():
    assert sentences("Lage\nDas Objekt liegt am Meer\n\n- Pool\n- Garten") == [
        "Lage",
        "Das Objekt liegt am Meer",
        "- Pool",
        "- Garten",
    ]


def test_lowercase_continuation_is_not_a_boundary():
    assert len(sentences("Siehe Abb. drei und mehr.")) == 1


def test_spans_cover_all_non_whitespace():
    text = "  Erster Satz.  Zweiter Satz?\n\nDritter »Satz«. Vierter "
    spans = split_sentences(text, "de")
    covered = "".join(text[s:e] for s, e in spans)
    assert covered.replace(" ", "") == "".join(text.split())


def test_empty_text():
    assert split_sentences("   \n ", "de") == []
