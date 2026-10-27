# Stage-1 experiments

37 documents, summaries from `summaries-groq_openai_gpt-oss-120b.json`, embeddings `BAAI/bge-large-en-v1.5`. Factors tuned one after another on 300 questions; the chosen settings are then scored on 300 disjoint questions.

## Tuning (seed-0 questions)

| Setting | Chunks | Mean words | Hits@1 | Hits@5 | Hits@20 | Hits@200 words | Hits@500 words |
|---|---|---|---|---|---|---|---|
| Fixed-size 100 words | 2322 | 84.6 | 74.7 | 93.0 | 97.3 | 85.0 | 93.7 |
| PIC, paper setting | 3331 | 58.9 | 68.0 | 89.7 | 97.3 | 76.7 | 87.7 |
| PIC, cap 100 / min 50 (0.2 baseline) | 2674 | 73.4 | 76.3 | 93.7 | 97.7 | 87.3 | 95.0 |
| smoothing window 1 | 2674 | 73.4 | 76.3 | 93.7 | 97.7 | 87.3 | 95.0 |
| smoothing window 3 | 2610 | 75.2 | 73.7 | 93.0 | 97.0 | 86.3 | 93.3 |
| smoothing window 5 | 2554 | 76.9 | 72.0 | 91.7 | 97.7 | 83.3 | 92.3 |
| threshold mean | 2674 | 73.4 | 76.3 | 93.7 | 97.7 | 87.3 | 95.0 |
| threshold median | 2681 | 73.2 | 75.7 | 93.7 | 97.7 | 88.3 | 95.3 |
| threshold mean-0.5sd | 2652 | 74.0 | 75.3 | 92.0 | 96.7 | 86.0 | 93.3 |
| threshold mean+0.5sd | 2686 | 73.1 | 73.7 | 93.0 | 97.3 | 86.0 | 93.3 |
| cap 60 / min 0 | 5311 | 37.0 | 72.0 | 90.3 | 97.0 | 89.7 | 95.0 |
| cap 60 / min 25 | 4430 | 44.3 | 74.0 | 93.0 | 97.7 | 91.7 | 95.7 |
| cap 60 / min 50 | 4345 | 45.2 | 73.7 | 93.0 | 98.0 | 91.3 | 95.3 |
| cap 80 / min 0 | 4552 | 43.1 | 72.7 | 91.7 | 98.3 | 89.7 | 95.3 |
| cap 80 / min 25 | 3665 | 53.6 | 74.7 | 93.0 | 98.0 | 88.3 | 95.7 |
| cap 80 / min 50 | 3336 | 58.9 | 74.7 | 93.3 | 98.3 | 87.3 | 95.3 |
| cap 80 / min 75 | 3336 | 58.9 | 74.7 | 93.3 | 98.3 | 87.3 | 95.3 |
| cap 100 / min 0 | 4100 | 47.9 | 74.3 | 91.3 | 97.7 | 87.0 | 93.7 |
| cap 100 / min 25 | 3224 | 60.9 | 74.7 | 92.3 | 97.3 | 88.3 | 94.7 |
| cap 100 / min 50 | 2681 | 73.2 | 75.7 | 93.7 | 97.7 | 88.3 | 95.3 |
| cap 100 / min 75 | 2680 | 73.3 | 75.7 | 93.7 | 97.7 | 88.3 | 95.3 |
| cap 120 / min 0 | 3838 | 51.2 | 71.0 | 91.7 | 98.0 | 85.3 | 93.7 |
| cap 120 / min 25 | 2969 | 66.1 | 72.0 | 93.0 | 97.3 | 86.7 | 93.7 |
| cap 120 / min 50 | 2326 | 84.4 | 71.3 | 93.0 | 97.7 | 83.3 | 93.0 |
| cap 120 / min 75 | 2263 | 86.8 | 71.7 | 93.0 | 97.7 | 83.3 | 93.0 |
| cap 150 / min 0 | 3618 | 54.3 | 70.7 | 90.7 | 97.0 | 83.3 | 92.3 |
| cap 150 / min 25 | 2759 | 71.2 | 72.0 | 92.0 | 96.7 | 83.3 | 91.7 |
| cap 150 / min 50 | 2017 | 97.3 | 72.7 | 92.0 | 96.7 | 78.7 | 91.7 |
| cap 150 / min 75 | 1821 | 107.8 | 72.7 | 91.0 | 96.7 | 76.0 | 90.7 |
| no context | 4430 | 44.3 | 74.0 | 93.0 | 97.7 | 91.7 | 95.7 |
| summary sentence as context | 4430 | 44.3 | 64.3 | 91.0 | 96.3 | 87.7 | 94.3 |
| Fixed-size + summary context | 2322 | 84.6 | 68.0 | 89.0 | 97.0 | 77.3 | 89.0 |
| relevance bonus 0.0 | 4430 | 44.3 | 74.0 | 93.0 | 97.7 | 91.7 | 95.7 |
| relevance bonus 0.01 | 4430 | 44.3 | 74.7 | 93.0 | 97.7 | 91.7 | 95.3 |
| relevance bonus 0.02 | 4430 | 44.3 | 74.3 | 92.3 | 97.3 | 90.7 | 95.7 |
| relevance bonus 0.04 | 4430 | 44.3 | 72.0 | 92.0 | 96.7 | 89.7 | 95.3 |

Chosen on tuning questions: window 1, threshold median, cap 60 / min 25 words, context off, bonus 0.0.

## Test (disjoint seed-0 questions 301-600)

| Setting | Chunks | Mean words | Hits@1 | Hits@5 | Hits@20 | Hits@200 words | Hits@500 words |
|---|---|---|---|---|---|---|---|
| Fixed-size 100 words | 2322 | 84.6 | 74.7 | 91.0 | 97.0 | 84.3 | 91.3 |
| Fixed-size + summary context | 2322 | 84.6 | 66.7 | 88.3 | 96.0 | 79.3 | 88.7 |
| Fixed-size 60 words (same cap) | 4135 | 47.5 | 74.7 | 91.0 | 95.7 | 89.0 | 93.7 |
| Sentence (fixed-size 1 word = 1 sentence) | 7834 | 25.1 | 65.7 | 87.3 | 94.3 | 89.0 | 94.3 |
| PIC, paper setting | 3331 | 58.9 | 63.7 | 87.3 | 95.0 | 71.7 | 84.7 |
| PIC, cap 100 / min 50 (0.2 baseline) | 2674 | 73.4 | 70.7 | 92.3 | 97.7 | 83.7 | 93.0 |
| PIC, chosen settings | 4430 | 44.3 | 74.0 | 90.7 | 95.7 | 88.7 | 94.3 |

McNemar's exact test on the test questions (questions only A hits / only B hits, p):

- hits@200w: chosen PIC vs fixed-size: 24 / 11, p = 0.041
- hits@200w: chosen PIC vs fixed-size 60 words: 5 / 6, p = 1.000
- hits@200w: chosen PIC vs sentence: 16 / 17, p = 1.000
- hits@200w: chosen PIC vs fixed-size + context: 41 / 13, p = 0.000
- hits@200w: chosen PIC vs PIC 0.2 baseline: 23 / 8, p = 0.011
- hits@200w: chosen PIC vs PIC paper: 62 / 11, p = 0.000
- hits@5: chosen PIC vs fixed-size: 10 / 11, p = 1.000
- hits@5: chosen PIC vs fixed-size 60 words: 4 / 5, p = 1.000
- hits@5: chosen PIC vs sentence: 24 / 14, p = 0.143
- hits@5: chosen PIC vs fixed-size + context: 23 / 16, p = 0.337
- hits@5: chosen PIC vs PIC 0.2 baseline: 9 / 14, p = 0.405
- hits@5: chosen PIC vs PIC paper: 24 / 14, p = 0.143
