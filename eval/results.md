| Method | Chunks | Mean words | Hits@1 | Hits@5 | Hits@20 | Hits@200 words | Hits@500 words |
|---|---|---|---|---|---|---|---|
| Sentence | 7834 | 25.1 | 68.7 | 86.3 | 96.3 | 88.3 | 95.3 |
| Paragraph | 1592 | 123.3 | 76.0 | 94.3 | 98.7 | 79.7 | 92.0 |
| Fixed-size (100 words) | 2322 | 84.6 | 69.7 | 90.0 | 96.3 | 78.3 | 90.7 |
| PIC (paper: no length limits) | 3331 | 58.9 | 66.7 | 90.3 | 96.7 | 74.0 | 88.0 |
| PIC, merged to ≥ 50 words | 1686 | 116.4 | 64.0 | 89.0 | 96.7 | 66.7 | 85.0 |
| PIC, 100-word cap, merged ≥ 50 | 2673 | 73.4 | 71.3 | 90.0 | 95.7 | 83.0 | 91.0 |

300 questions (seed 0), 37 documents, LLM `groq/openai/gpt-oss-120b`, embeddings `BAAI/bge-large-en-v1.5`, summary input capped at 26,000 characters. PIC marked 50.3% of its chunks relevant; 67.7% of the chunks retrieved in the top 5 were relevant ones.
