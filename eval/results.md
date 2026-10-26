| Method | Chunks | Mean words | Hits@1 | Hits@5 | Hits@20 | Hits@200 words | Hits@500 words |
|---|---|---|---|---|---|---|---|
| Sentence | 2486 | 23.7 | 63.7 | 86.7 | 95.7 | 90.0 | 95.0 |
| Paragraph | 574 | 102.4 | 71.3 | 94.3 | 99.0 | 77.0 | 92.3 |
| Fixed-size (100 words) | 687 | 85.6 | 65.3 | 89.7 | 98.0 | 82.0 | 90.3 |
| PIC (paper: no length limits) | 1078 | 54.5 | 64.0 | 87.7 | 95.7 | 72.0 | 84.0 |
| PIC, merged to ≥ 50 words | 516 | 113.9 | 66.0 | 88.3 | 97.7 | 68.3 | 84.0 |
| PIC, 100-word cap, merged ≥ 50 | 814 | 72.2 | 71.3 | 90.3 | 98.3 | 83.0 | 91.3 |

300 questions (seed 0), 11 documents, LLM `gemini/gemini-3.5-flash`, embeddings `BAAI/bge-large-en-v1.5`. PIC marked 50.2% of its chunks relevant; 69.4% of the chunks retrieved in the top 5 were relevant ones.
