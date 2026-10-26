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


def test_abbreviations_follow_the_language():
    assert sentences("Die Wohnung Nr. 5 liegt z. B. am 3. Oktober frei. Danach nicht.") == [
        "Die Wohnung Nr. 5 liegt z. B. am 3. Oktober frei.",
        "Danach nicht.",
    ]
    assert sentences("Dr. Kovács a Kft. ügyvezetője. Ez jó.", "hu") == [
        "Dr. Kovács a Kft. ügyvezetője.",
        "Ez jó.",
    ]
    assert sentences("El Sr. García llegó. Se fue.", "es") == ["El Sr. García llegó.", "Se fue."]
    assert len(sentences("We sold 3. Then we stopped.", "en")) == 2


def test_lowercase_continuation_is_never_a_boundary():
    # Abbreviations sentencex does not know, including before non-ASCII letters.
    assert sentences("Siehe Abb. über uns. Ja.") == ["Siehe Abb. über uns.", "Ja."]
    assert sentences("Voir p. élevé. Oui.", "fr") == ["Voir p. élevé.", "Oui."]


def test_line_breaks_end_sentences():
    assert sentences("Lage\nDas Objekt liegt am Meer\n\n- Pool\n- Garten") == [
        "Lage",
        "Das Objekt liegt am Meer",
        "- Pool",
        "- Garten",
    ]


def test_scripts_without_spaces_after_terminators():
    assert sentences("東京は大きい。大阪も「大きい！」京都は？", "ja") == [
        "東京は大きい。",
        "大阪も「大きい！」",
        "京都は？",
    ]
    assert len(sentences("هل أنت بخير؟ نعم.", "ar")) == 2
    assert len(sentences("यह घर है। वह बड़ा है।", "hi")) == 2


def test_spans_cover_all_non_whitespace():
    text = "  Erster Satz.  Zweiter Satz?\n\nDritter »Satz«. Vierter "
    spans = split_sentences(text, "de")
    assert all(text[s:e] == text[s:e].strip() for s, e in spans)
    covered = "".join(text[s:e] for s, e in spans)
    assert covered.replace(" ", "") == "".join(text.split())


def test_empty_text():
    assert split_sentences("   \n ", "de") == []
    assert split_sentences("", "de") == []
