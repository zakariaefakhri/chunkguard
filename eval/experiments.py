"""Stage-1 experiments: what improves PIC without extra LLM calls?

Factors: smoothing the similarity curve, the threshold statistic, the length limits,
context in the chunk vector (first sentence of the summary prepended before embedding),
and a retrieval bonus for chunks flagged relevant.

Honest selection: factors are tuned one after another on 300 tuning questions (seed 0).
The final settings are then scored on 300 disjoint test questions against fixed-size
chunking, with McNemar's exact test on the paired per-question hits.

Primary metric: hits within a 200-word budget (fair across chunk sizes); Hits@5 second.
Uses only the summaries run.py cached; no LLM calls.

    uv run --with numpy --with sentence-transformers python eval/experiments.py
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import numpy as np
from run import CACHE, EmbeddingCache, fixed_chunks, load_squad, words

from chunkguard import split_sentences
from chunkguard.pic import _merge_short, _runs, _split_long

BUDGETS = (200, 500)
KS = (1, 5, 20)
PRIMARY, SECONDARY = "hits@200w", "hits@5"


def smooth(sims: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return sims
    padded = np.pad(sims, window // 2, mode="edge")
    return np.convolve(padded, np.ones(window) / window, mode="valid")


THRESHOLDS = {
    "mean": lambda s: float(np.mean(s)),
    "median": lambda s: float(np.median(s)),
    "mean-0.5sd": lambda s: float(np.mean(s) - 0.5 * np.std(s)),
    "mean+0.5sd": lambda s: float(np.mean(s) + 0.5 * np.std(s)),
}


class Doc:
    def __init__(self, title, text, summary, cache):
        self.title = title
        self.spans = split_sentences(text, "en")
        self.sentences = [text[s:e] for s, e in self.spans]
        self.text = text
        self.sizes = [words(s) for s in self.sentences]
        vectors = cache([summary, *self.sentences])
        self.sims = vectors[1:] @ vectors[0]
        first = split_sentences(summary, "en")
        self.context = summary[first[0][0] : first[0][1]] if first else summary


def pic_chunks(doc: Doc, window=1, threshold="mean", max_len=None, min_len=0):
    sims = smooth(doc.sims, window)
    t = THRESHOLDS[threshold](sims)
    cap = math.inf if max_len is None else max_len
    groups = _runs([bool(x >= t) for x in sims])
    groups = [p for g in groups for p in _split_long(g, doc.sizes, cap)]
    groups = _merge_short(groups, doc.sizes, list(map(float, sims)), min_len, cap)
    texts, flags = [], []
    for first, end in groups:
        texts.append(doc.text[doc.spans[first][0] : doc.spans[end - 1][1]])
        size = sum(doc.sizes[first:end])
        relevant = sum(
            n for n, x in zip(doc.sizes[first:end], sims[first:end], strict=False) if x >= t
        )
        flags.append(2 * relevant >= size)
    return texts, flags


def score(chunks, flags, contexts, questions, qvec, cache, bonus=0.0):
    """Per-question hit arrays for every metric."""
    texts = [c for doc in chunks for c in doc]
    lowered = [t.lower() for t in texts]
    embed_texts = [
        f"{ctx}\n\n{c}" if ctx else c
        for doc, ctx in zip(chunks, contexts, strict=True)
        for c in doc
    ]
    cvec = cache(embed_texts)
    sims = qvec @ cvec.T
    if bonus:
        sims = sims + bonus * np.array([f for doc in flags for f in doc], dtype=np.float32)
    order = np.argsort(-sims, axis=1)[:, :60]
    sizes = [words(t) for t in texts]
    out = {}
    for k in KS:
        out[f"hits@{k}"] = np.array(
            [
                any(any(a in lowered[j] for a in q["gold"]) for j in row[:k])
                for row, q in zip(order, questions, strict=False)
            ]
        )
    for budget in BUDGETS:
        hits = []
        for row, q in zip(order, questions, strict=False):
            total, found = 0, False
            for j in row:
                if total and total + sizes[j] > budget:
                    break
                total += sizes[j]
                found = found or any(a in lowered[j] for a in q["gold"])
            hits.append(found)
        out[f"hits@{budget}w"] = np.array(hits)
    out["chunks"] = len(texts)
    out["mean_words"] = round(sum(sizes) / len(sizes), 1)
    return out


def pct(hits: np.ndarray) -> float:
    return round(100 * hits.mean(), 1)


def mcnemar(a: np.ndarray, b: np.ndarray) -> tuple[int, int, float]:
    """Exact two-sided McNemar test: (a only, b only, p)."""
    only_a, only_b = int((a & ~b).sum()), int((~a & b).sum())
    n = only_a + only_b
    if n == 0:
        return only_a, only_b, 1.0
    tail = sum(math.comb(n, i) for i in range(min(only_a, only_b) + 1)) / 2**n
    return only_a, only_b, min(1.0, 2 * tail)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summaries", default="summaries-groq_openai_gpt-oss-120b.json")
    parser.add_argument("--embed", default="BAAI/bge-large-en-v1.5")
    parser.add_argument("--questions", type=int, default=300)
    args = parser.parse_args()

    from sentence_transformers import SentenceTransformer

    encoder = SentenceTransformer(args.embed)
    cache = EmbeddingCache(
        lambda texts: encoder.encode(texts, batch_size=64).tolist(),
        CACHE / f"embeddings-{args.embed.replace('/', '_')}.npz",
    )
    summaries = json.loads((CACHE / args.summaries).read_text())
    articles = [a for a in load_squad() if a["title"] in summaries]
    texts = {a["title"]: "\n\n".join(p["context"] for p in a["paragraphs"]) for a in articles}
    docs = [Doc(t, texts[t], summaries[t], cache) for t in texts]

    pool = [
        {"question": q["question"], "gold": {x["text"].lower().strip() for x in q["answers"]}}
        for a in articles
        for p in a["paragraphs"]
        for q in p["qas"]
    ]
    random.Random(0).shuffle(pool)
    tune, test = pool[: args.questions], pool[args.questions : 2 * args.questions]
    tune_vec = cache([q["question"] for q in tune])
    test_vec = cache([q["question"] for q in test])

    no_context = [""] * len(docs)
    summary_context = [d.context for d in docs]
    log: list[str] = []

    def run(name, settings, questions, qvec, context=no_context, bonus=0.0):
        chunks, flags = zip(*(pic_chunks(d, **settings) for d in docs), strict=False)
        result = score(chunks, flags, context, questions, qvec, cache, bonus)
        line = (
            f"| {name} | {result['chunks']} | {result['mean_words']} | "
            + " | ".join(
                str(pct(result[m]))
                for m in [*(f"hits@{k}" for k in KS), *(f"hits@{b}w" for b in BUDGETS)]
            )
            + " |"
        )
        log.append(line)
        print(line, flush=True)
        return result

    def run_fixed(name, questions, qvec, context=no_context, size=100):
        chunks = [fixed_chunks(d.text, size) for d in docs]
        flags = [[False] * len(c) for c in chunks]
        result = score(chunks, flags, context, questions, qvec, cache)
        line = (
            f"| {name} | {result['chunks']} | {result['mean_words']} | "
            + " | ".join(
                str(pct(result[m]))
                for m in [*(f"hits@{k}" for k in KS), *(f"hits@{b}w" for b in BUDGETS)]
            )
            + " |"
        )
        log.append(line)
        print(line, flush=True)
        return result

    def best(candidates):
        return max(candidates, key=lambda c: (pct(c[1][PRIMARY]), pct(c[1][SECONDARY])))

    header = (
        "| Setting | Chunks | Mean words | Hits@1 | Hits@5 | Hits@20 "
        "| Hits@200 words | Hits@500 words |"
    )
    rule = "|---|---|---|---|---|---|---|---|"

    log += ["## Tuning (seed-0 questions)", "", header, rule]
    run_fixed("Fixed-size 100 words", tune, tune_vec)
    run("PIC, paper setting", dict(), tune, tune_vec)
    settings = dict(window=1, threshold="mean", max_len=100, min_len=50)
    run("PIC, cap 100 / min 50 (0.2 baseline)", settings, tune, tune_vec)

    candidates = [
        (w, run(f"smoothing window {w}", {**settings, "window": w}, tune, tune_vec))
        for w in (1, 3, 5)
    ]
    settings["window"] = best(candidates)[0]

    candidates = [
        (t, run(f"threshold {t}", {**settings, "threshold": t}, tune, tune_vec)) for t in THRESHOLDS
    ]
    settings["threshold"] = best(candidates)[0]

    candidates = []
    for cap in (60, 80, 100, 120, 150):
        for low in (0, 25, 50, 75):
            if low < cap:
                s = {**settings, "max_len": cap, "min_len": low}
                candidates.append(((cap, low), run(f"cap {cap} / min {low}", s, tune, tune_vec)))
    settings["max_len"], settings["min_len"] = best(candidates)[0]

    candidates = [
        (False, run("no context", settings, tune, tune_vec)),
        (True, run("summary sentence as context", settings, tune, tune_vec, summary_context)),
    ]
    use_context = best(candidates)[0]
    context = summary_context if use_context else no_context
    run_fixed("Fixed-size + summary context", tune, tune_vec, summary_context)

    candidates = [
        (b, run(f"relevance bonus {b}", settings, tune, tune_vec, context, b))
        for b in (0.0, 0.01, 0.02, 0.04)
    ]
    bonus = best(candidates)[0]

    chosen = (
        f"window {settings['window']}, threshold {settings['threshold']}, "
        f"cap {settings['max_len']} / min {settings['min_len']} words, "
        f"context {'on' if use_context else 'off'}, bonus {bonus}"
    )
    print("chosen:", chosen, flush=True)

    log += [
        "",
        f"Chosen on tuning questions: {chosen}.",
        "",
        "## Test (disjoint seed-0 questions 301-600)",
        "",
        header,
        rule,
    ]
    fixed = run_fixed("Fixed-size 100 words", test, test_vec)
    fixed_ctx = run_fixed("Fixed-size + summary context", test, test_vec, summary_context)
    # Short chunks win word budgets by themselves, so compare with fixed-size at the same cap
    # and with one sentence per chunk.
    cap = settings["max_len"]
    fixed_same = run_fixed(f"Fixed-size {cap} words (same cap)", test, test_vec, size=cap)
    sentence = run_fixed("Sentence (fixed-size 1 word = 1 sentence)", test, test_vec, size=1)
    paper = run("PIC, paper setting", dict(), test, test_vec)
    base = run(
        "PIC, cap 100 / min 50 (0.2 baseline)",
        dict(window=1, threshold="mean", max_len=100, min_len=50),
        test,
        test_vec,
    )
    final = run("PIC, chosen settings", settings, test, test_vec, context, bonus)

    log += [
        "",
        "McNemar's exact test on the test questions (questions only A hits / only B hits, p):",
        "",
    ]
    for metric in (PRIMARY, SECONDARY):
        for name, other in (
            ("fixed-size", fixed),
            (f"fixed-size {cap} words", fixed_same),
            ("sentence", sentence),
            ("fixed-size + context", fixed_ctx),
            ("PIC 0.2 baseline", base),
            ("PIC paper", paper),
        ):
            a_only, b_only, p = mcnemar(final[metric], other[metric])
            line = f"- {metric}: chosen PIC vs {name}: {a_only} / {b_only}, p = {p:.3f}"
            log.append(line)
            print(line, flush=True)

    out = Path(__file__).parent / "experiments.md"
    out.write_text(
        "# Stage-1 experiments\n\n"
        f"{len(docs)} documents, summaries from `{args.summaries}`, embeddings `{args.embed}`. "
        "Factors tuned one after another on 300 questions; the chosen settings are then "
        "scored on 300 disjoint questions.\n\n" + "\n".join(log) + "\n"
    )


if __name__ == "__main__":
    main()
