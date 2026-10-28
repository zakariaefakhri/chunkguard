# Sentence boundaries: fixed list against learned abbreviations

UD German-GSD and English-EWT, original text, sentences joined into blocks of 20, two-fold
cross-validation for the learned abbreviations (Punkt, Kiss and Strunk 2006, via NLTK).
"chunkguard (fixed list)" is the splitter after the fixes below; the released 0.3.0 had an
English recall of 70.59 (F1 81.94), because its lowercase and ordinal rules dropped 562
boundaries that sentencex finds ("blog. hey that's", "the age of 13. Police said").

Findings:

- Learned abbreviations change almost nothing on either treebank (F1 within 0.05). They are
  not in the library; the learning code stays in this script.
- In German, chunkguard trails sentencex by about 100 boundaries after short numbers. Most
  are faulty gold sentences in GSD that end after an ordinal ("ging Ruge am 1." followed by
  "Doch kurz vor ..."), so this gap overstates the cost of the ordinal rule.
- NLTK's Punkt, trained on the same data, has the best F1 in both languages, mainly through
  recall; in German it partly learns the GSD gold errors. Most boundaries every method misses
  have no sentence punctuation at all (headlines, signatures).
- The business-German cases come from the code review that motivated the fixed list, so the
  list scores 10 of 10 by construction; they are regression checks, not an evaluation.

### de: 14050 gold boundaries in 740 blocks

Learned per fold (all / ≤ 6 / ≤ 4 letters): [(55, 54, 52), (60, 56, 55)]

| Method | Precision | Recall | F1 | False cuts | Missed |
|---|---|---|---|---|---|
| sentencex alone | 99.40 | 90.53 | 94.76 | 77 | 1331 |
| NLTK Punkt alone (trained) | 98.60 | 97.03 | 97.81 | 193 | 417 |
| chunkguard (fixed list) | 99.50 | 89.82 | 94.41 | 64 | 1430 |
| learned, all | 99.50 | 89.84 | 94.43 | 63 | 1427 |
| learned, ≤ 6 letters | 99.50 | 89.87 | 94.44 | 63 | 1423 |
| learned, ≤ 4 letters | 99.50 | 89.87 | 94.44 | 63 | 1423 |
| fixed list + learned ≤ 6 | 99.50 | 89.80 | 94.40 | 63 | 1433 |

### en: 13889 gold boundaries in 732 blocks

Learned per fold (all / ≤ 6 / ≤ 4 letters): [(57, 56, 56), (70, 69, 67)]

| Method | Precision | Recall | F1 | False cuts | Missed |
|---|---|---|---|---|---|
| sentencex alone | 97.16 | 74.63 | 84.42 | 303 | 3523 |
| NLTK Punkt alone (trained) | 96.53 | 78.37 | 86.51 | 391 | 3004 |
| chunkguard (fixed list) | 97.21 | 74.63 | 84.44 | 297 | 3523 |
| learned, all | 97.33 | 74.52 | 84.41 | 284 | 3539 |
| learned, ≤ 6 letters | 97.33 | 74.52 | 84.41 | 284 | 3539 |
| learned, ≤ 4 letters | 97.33 | 74.53 | 84.42 | 284 | 3538 |
| fixed list + learned ≤ 6 | 97.34 | 74.52 | 84.41 | 283 | 3539 |

### Business-German cases

| Method | Correct |
|---|---|
| sentencex alone | 3 / 10 |
| chunkguard (fixed list) | 10 / 10 |
| learned from GSD, ≤ 6 letters | 5 / 10 |
