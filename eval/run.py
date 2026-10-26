"""Retrieval evaluation: PIC against sentence, paragraph and fixed-size chunking.

Corpus: the 48 Wikipedia articles of the SQuAD v1.1 dev set (one article = one
document, paragraphs joined). Questions: a seeded sample of the 10,570 SQuAD
questions; a chunk is a hit when it contains one of the gold answer strings.
Every method embeds its chunks with the same model, questions are embedded once,
and Hits@k is the share of questions with a hit among the k nearest chunks.

    uv run --with numpy --with sentence-transformers python eval/run.py --env .env

The summarizer comes from .env (litellm extra). Embeddings run locally with
sentence-transformers (default BAAI/bge-large-en-v1.5, the paper's model), or
with the .env embedding model via --embed env. Summaries and embeddings are
cached under eval/cache so a rerun with other settings costs nothing.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np

from chunkguard import PICChunker, split_sentences
from chunkguard.backend import _parts, litellm_complete, litellm_embed

SQUAD_URL = "https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json"
CACHE = Path(__file__).parent / "cache"
FIXED_WORDS = 100  # as in the paper's fixed-size baseline
KS = (1, 5, 20)
# Hits@k favours long chunks (more text per retrieved item), so we also score the chunks
# that fit into a fixed budget of retrieved words, which compares methods on equal footing.
WORD_BUDGETS = (200, 500)


def words(text: str) -> int:
    return len(text.split())


def load_squad() -> list[dict]:
    path = CACHE / "dev-v1.1.json"
    if not path.exists():
        CACHE.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(SQUAD_URL, path)
    return json.loads(path.read_text())["data"]


def sentence_chunks(text: str) -> list[str]:
    return [text[s:e] for s, e in split_sentences(text, "en")]


def fixed_chunks(text: str, words: int = FIXED_WORDS) -> list[str]:
    """Up to `words` words per chunk, never cutting a sentence."""
    chunks, current, count = [], [], 0
    for sentence in sentence_chunks(text):
        n = len(sentence.split())
        if current and count + n > words:
            chunks.append(" ".join(current))
            current, count = [], 0
        current.append(sentence)
        count += n
    if current:
        chunks.append(" ".join(current))
    return chunks


class EmbeddingCache:
    """sha1(text) -> float16 vector, persisted as one .npz per run directory."""

    def __init__(self, embed, path: Path) -> None:
        self.embed = embed
        self.path = path
        self.vectors: dict[str, np.ndarray] = {}
        if path.exists():
            with np.load(path) as stored:
                self.vectors = {key: stored[key] for key in stored.files}

    def __call__(self, texts: list[str]) -> np.ndarray:
        keys = [hashlib.sha1(t.encode()).hexdigest() for t in texts]
        missing = sorted(
            {k: t for k, t in zip(keys, texts, strict=True) if k not in self.vectors}.items()
        )
        for i in range(0, len(missing), 512):
            batch = missing[i : i + 512]
            vectors = self.embed([t for _, t in batch])
            for (key, _), vector in zip(batch, vectors, strict=True):
                self.vectors[key] = np.asarray(vector, dtype=np.float16)
            print(f"  embedded {min(i + 512, len(missing))}/{len(missing)} new texts", flush=True)
        if missing:
            np.savez(self.path, **self.vectors)
        matrix = np.stack([self.vectors[k] for k in keys]).astype(np.float32)
        return matrix / np.linalg.norm(matrix, axis=1, keepdims=True)


def evaluate(name, chunks_per_doc, gold, question_vectors, cache, relevant_flags=None) -> dict:
    texts = [c for doc in chunks_per_doc for c in doc]
    lowered = [t.lower() for t in texts]
    print(f"{name}: {len(texts)} chunks", flush=True)
    chunk_vectors = cache(texts)
    order = np.argsort(-(question_vectors @ chunk_vectors.T), axis=1)[:, : max(KS)]
    result = {
        "chunks": len(texts),
        "mean_words": round(sum(len(t.split()) for t in texts) / len(texts), 1),
    }
    for k in KS:
        hit = [
            any(any(a in lowered[j] for a in answers) for j in row[:k])
            for row, answers in zip(order, gold, strict=True)
        ]
        result[f"hits@{k}"] = round(100 * sum(hit) / len(hit), 1)
    sizes = [words(t) for t in texts]
    for budget in WORD_BUDGETS:
        hit = []
        for row, answers in zip(order, gold, strict=True):
            total, found = 0, False
            for j in row:
                if total and total + sizes[j] > budget:
                    break
                total += sizes[j]
                found = found or any(a in lowered[j] for a in answers)
            hit.append(found)
        result[f"hits@{budget}w"] = round(100 * sum(hit) / len(hit), 1)
    if relevant_flags is not None:
        flags = [f for doc in relevant_flags for f in doc]
        result["relevant_share"] = round(100 * sum(flags) / len(flags), 1)
        retrieved = [flags[j] for row in order for j in row[:5]]
        result["relevant_among_top5"] = round(100 * sum(retrieved) / len(retrieved), 1)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default=".env")
    parser.add_argument("--questions", type=int, default=300)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--embed", default="BAAI/bge-large-en-v1.5", help="HF model, or 'env'")
    parser.add_argument("--llm-pause", type=float, default=6, help="seconds between LLM calls")
    parser.add_argument(
        "--only-cached", action="store_true", help="use only documents with a cached summary"
    )
    args = parser.parse_args()

    settings, llm, embedder, _ = _parts(args.env)
    complete = litellm_complete(settings.llm_model, **llm)
    if args.embed == "env":
        embedding_model = settings.embedding_model
        embed = litellm_embed(embedding_model, settings.embedding_batch_size, **embedder)
    else:
        from sentence_transformers import SentenceTransformer

        embedding_model = args.embed
        encoder = SentenceTransformer(embedding_model)
        embed = lambda texts: encoder.encode(texts, batch_size=64).tolist()  # noqa: E731
    cache = EmbeddingCache(embed, CACHE / f"embeddings-{embedding_model.replace('/', '_')}.npz")

    summaries_path = CACHE / f"summaries-{settings.llm_model.replace('/', '_')}.json"
    summaries = json.loads(summaries_path.read_text()) if summaries_path.exists() else {}
    articles = load_squad()
    if args.only_cached:
        articles = [a for a in articles if a["title"] in summaries]
    documents = ["\n\n".join(p["context"] for p in a["paragraphs"]) for a in articles]
    paragraphs = [[p["context"] for p in a["paragraphs"]] for a in articles]

    rng = random.Random(args.seed)
    all_questions = [q for a in articles for p in a["paragraphs"] for q in p["qas"]]
    questions = rng.sample(all_questions, args.questions)
    gold = [{ans["text"].lower().strip() for ans in q["answers"]} for q in questions]
    question_vectors = cache([q["question"] for q in questions])

    variants = {
        "PIC (paper: no length limits)": dict(max_length=None, min_length=0),
        "PIC, merged to ≥ 50 words": dict(max_length=None, min_length=50, length=words),
        f"PIC, {FIXED_WORDS}-word cap, merged ≥ 50": dict(
            max_length=FIXED_WORDS, min_length=50, length=words
        ),
    }
    results = {
        "Sentence": evaluate(
            "Sentence", [sentence_chunks(d) for d in documents], gold, question_vectors, cache
        ),
        "Paragraph": evaluate("Paragraph", paragraphs, gold, question_vectors, cache),
        f"Fixed-size ({FIXED_WORDS} words)": evaluate(
            "Fixed-size", [fixed_chunks(d) for d in documents], gold, question_vectors, cache
        ),
    }
    started = time.time()
    for name, options in variants.items():
        chunker = PICChunker(complete, embed, **options)
        pic_chunks, pic_flags = [], []
        for article, text in zip(articles, documents, strict=True):
            title = article["title"]
            if title not in summaries and started + 1 < time.time():
                time.sleep(args.llm_pause)  # free tiers allow few LLM requests per minute
            chunks = chunker.split(title, text, "en", summary=summaries.get(title))
            summaries[title] = chunks[0].summary
            summaries_path.write_text(json.dumps(summaries, ensure_ascii=False, indent=1))
            pic_chunks.append([c.text for c in chunks])
            pic_flags.append([c.relevant for c in chunks])
        print(f"{name}: summaries done ({time.time() - started:.0f}s)", flush=True)
        results[name] = evaluate(name, pic_chunks, gold, question_vectors, cache, pic_flags)

    columns = [f"hits@{k}" for k in KS] + [f"hits@{b}w" for b in WORD_BUDGETS]
    header = (
        "| Method | Chunks | Mean words | "
        + " | ".join([f"Hits@{k}" for k in KS] + [f"Hits@{b} words" for b in WORD_BUDGETS])
        + " |"
    )
    lines = [header, "|" + "---|" * (3 + len(columns))]
    for name, r in results.items():
        lines.append(
            f"| {name} | {r['chunks']} | {r['mean_words']} | "
            + " | ".join(f"{r[c]}" for c in columns)
            + " |"
        )
    table = "\n".join(lines)
    pic = results["PIC (paper: no length limits)"]
    notes = (
        f"\n\n{args.questions} questions (seed {args.seed}), {len(articles)} documents, "
        f"LLM `{settings.llm_model}`, embeddings `{embedding_model}`. "
        f"PIC marked {pic['relevant_share']}% of its chunks relevant; "
        f"{pic['relevant_among_top5']}% of the chunks retrieved in the top 5 were relevant ones.\n"
    )
    print("\n" + table + notes)
    (Path(__file__).parent / "results.md").write_text(table + notes)
    (Path(__file__).parent / "results.json").write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    sys.exit(main())
