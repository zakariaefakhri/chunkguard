"""Sentence boundary evaluation on Universal Dependencies treebanks.

Does learning abbreviations from the corpus (Punkt) beat the fixed list? Gold
sentences come from UD German-GSD and English-EWT (train + test, original text).
Consecutive sentences are joined with a space into blocks of 20; the boundaries
between them are gold. Two-fold cross-validation: abbreviations are learned on one
half of the blocks and measured on the other, then the other way round.

    uv run --with nltk python eval/sentences.py
"""

from __future__ import annotations

import urllib.request
from collections.abc import Iterable
from pathlib import Path

import sentencex
from nltk.tokenize.punkt import PunktSentenceTokenizer, PunktTrainer

import chunkguard.sentences as sentences_module
from chunkguard import split_sentences

CACHE = Path(__file__).parent / "cache"
UD = "https://raw.githubusercontent.com/UniversalDependencies"
TREEBANKS = {
    "de": (
        "UD_German-GSD/master/de_gsd-ud-train.conllu",
        "UD_German-GSD/master/de_gsd-ud-test.conllu",
    ),
    "en": (
        "UD_English-EWT/master/en_ewt-ud-train.conllu",
        "UD_English-EWT/master/en_ewt-ud-test.conllu",
    ),
}
BLOCK = 20

# Business-German cases from the code review and the 0.3.0 tests, with their gold split.
CASES = [
    ("Wir liefern an die Fa. Rewe in Köln. Danach ist Schluss.", 2),
    ("Rufen Sie an unter Tel. 069 123. Wir sind montags erreichbar.", 2),
    ("Das Büro liegt in der Hauptstr. 5. Parkplätze gibt es keine.", 2),
    ("Es ist Zimmer Nr. 5. Danach links.", 2),
    ("Ein Zimmer aus dem 18. Jahrhundert. Wir wohnen im 3. Stock. Laut.", 3),
    ("Der 3. Platz ging an Anna. Sie freute sich.", 2),
    ("Treffen am 5. Okt. Bitte pünktlich.", 2),
    ("Preis inkl. Frühstück. Danach Abreise.", 2),
    ("Die Lieferung erfolgt zzgl. Versand. Bezahlung per Rechnung.", 2),
    ("Herausgegeben von Gebr. Grimm. Erschienen 1812.", 2),
]


def learn_abbreviations(texts: Iterable[str], *, max_letters: int = 6) -> frozenset[str]:
    """Return the abbreviations the Punkt algorithm finds in ``texts``.

    Punkt (Kiss and Strunk, 2006) needs no labelled data: a word with a final
    period that is often followed by a lowercase word, or rarely occurs without
    the period, counts as an abbreviation. Learn from documents of the kind you
    will chunk, so their own abbreviations ("Flst.", "lfd.") are covered.

    Words longer than ``max_letters`` letters are dropped; on small corpora
    Punkt also picks up ordinary words that happened to end a sentence once.
    Pass the result to ``PICChunker(abbreviations=...)`` or ``split_sentences``.
    """
    trainer = PunktTrainer()
    for text in texts:
        trainer.train(text, finalize=False)
    trainer.finalize_training()
    return frozenset(
        word
        for word in trainer.get_params().abbrev_types
        if sum(c.isalpha() for c in word) <= max_letters
    )


def sentences_of(path: Path) -> list[str]:
    return [
        line[9:].strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.startswith("# text = ")
    ]


def load(language: str) -> list[str]:
    out = []
    for name in TREEBANKS[language]:
        path = CACHE / "ud" / name.rsplit("/", 1)[1]
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            urllib.request.urlretrieve(f"{UD}/{name}", path)
        out += sentences_of(path)
    return out


def blocks(sents: list[str]) -> list[tuple[str, set[int]]]:
    out = []
    for i in range(0, len(sents), BLOCK):
        part = sents[i : i + BLOCK]
        text, gold, pos = "", set(), 0
        for k, s in enumerate(part):
            text += s
            pos += len(s)
            if k < len(part) - 1:
                gold.add(pos)
                text += " "
                pos += 1
        out.append((text, gold))
    return out


def predicted(spans: list[tuple[int, int]]) -> set[int]:
    return {end for _, end in spans[:-1]}


def score(counts: dict, gold: set[int], pred: set[int]) -> None:
    counts["tp"] += len(gold & pred)
    counts["fp"] += len(pred - gold)
    counts["fn"] += len(gold - pred)


def sentencex_spans(text: str, language: str) -> list[tuple[int, int]]:
    spans = []
    for b in sentencex.get_sentence_boundaries(language, text):
        piece = text[b["start_index"] : b["end_index"]]
        left = len(piece) - len(piece.lstrip())
        if piece.strip():
            spans.append((b["start_index"] + left, b["start_index"] + len(piece.rstrip())))
    return spans


def without_fixed_list(fn):
    """Run fn with chunkguard's hand-written abbreviation list switched off."""
    saved = sentences_module._EXTRA_ABBREVIATIONS, sentences_module._ABBREVIATION_SUFFIXES
    sentences_module._EXTRA_ABBREVIATIONS, sentences_module._ABBREVIATION_SUFFIXES = {}, {}
    sentences_module._ends_sentence.cache_clear()
    try:
        return fn()
    finally:
        sentences_module._EXTRA_ABBREVIATIONS, sentences_module._ABBREVIATION_SUFFIXES = saved
        sentences_module._ends_sentence.cache_clear()


def evaluate(language: str) -> tuple[list[str], dict]:
    data = blocks(load(language))
    folds = [[b for i, b in enumerate(data) if i % 2 == f] for f in (0, 1)]
    methods = [
        "sentencex alone",
        "NLTK Punkt alone (trained)",
        "chunkguard (fixed list)",
        "learned, all",
        "learned, ≤ 6 letters",
        "learned, ≤ 4 letters",
        "fixed list + learned ≤ 6",
    ]
    counts = {m: {"tp": 0, "fp": 0, "fn": 0} for m in methods}
    learned_sizes = []
    for f in (0, 1):
        train = [t for t, _ in folds[1 - f]]
        learned_all = learn_abbreviations(train, max_letters=1000)
        learned_6 = learn_abbreviations(train, max_letters=6)
        learned_4 = learn_abbreviations(train, max_letters=4)
        learned_sizes.append((len(learned_all), len(learned_6), len(learned_4)))
        trainer = PunktTrainer()
        for t in train:
            trainer.train(t, finalize=False)
        trainer.finalize_training()
        punkt = PunktSentenceTokenizer(trainer.get_params())
        for text, gold in folds[f]:
            score(counts["sentencex alone"], gold, predicted(sentencex_spans(text, language)))
            score(
                counts["NLTK Punkt alone (trained)"],
                gold,
                predicted(list(punkt.span_tokenize(text))),
            )
            score(
                counts["chunkguard (fixed list)"],
                gold,
                predicted(split_sentences(text, language)),
            )
            for name, learned in (
                ("learned, all", learned_all),
                ("learned, ≤ 6 letters", learned_6),
                ("learned, ≤ 4 letters", learned_4),
            ):
                spans = without_fixed_list(
                    lambda t=text, a=learned: split_sentences(t, language, abbreviations=a)
                )
                score(counts[name], gold, predicted(spans))
            spans = split_sentences(text, language, abbreviations=learned_6)
            score(counts["fixed list + learned ≤ 6"], gold, predicted(spans))
    lines = [
        f"### {language}: {sum(len(g) for _, g in data)} gold boundaries in {len(data)} blocks",
        "",
        f"Learned per fold (all / ≤ 6 / ≤ 4 letters): {learned_sizes}",
        "",
        "| Method | Precision | Recall | F1 | False cuts | Missed |",
        "|---|---|---|---|---|---|",
    ]
    for m in methods:
        c = counts[m]
        p = c["tp"] / (c["tp"] + c["fp"])
        r = c["tp"] / (c["tp"] + c["fn"])
        lines.append(
            f"| {m} | {100 * p:.2f} | {100 * r:.2f} | {100 * 2 * p * r / (p + r):.2f} "
            f"| {c['fp']} | {c['fn']} |"
        )
    return lines, counts


def cases() -> list[str]:
    learned = learn_abbreviations([t for t, _ in blocks(load("de"))], max_letters=6)
    lines = ["### Business-German cases", "", "| Method | Correct |", "|---|---|"]
    for name, fn in (
        ("sentencex alone", lambda t: sentencex_spans(t, "de")),
        ("chunkguard (fixed list)", lambda t: split_sentences(t, "de")),
        (
            "learned from GSD, ≤ 6 letters",
            lambda t: without_fixed_list(lambda: split_sentences(t, "de", abbreviations=learned)),
        ),
    ):
        ok = sum(len(fn(text)) == n for text, n in CASES)
        lines.append(f"| {name} | {ok} / {len(CASES)} |")
    return lines


def main() -> None:
    report = ["# Sentence boundaries: fixed list against learned abbreviations", ""]
    for language in ("de", "en"):
        lines, _ = evaluate(language)
        report += lines + [""]
        print("\n".join(lines), flush=True)
    report += cases()
    print("\n".join(cases()))
    (Path(__file__).parent / "sentences.md").write_text("\n".join(report) + "\n")


if __name__ == "__main__":
    main()
