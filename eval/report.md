# DocQA evaluation

30 documents and 40 questions (28 answerable, 8 unanswerable and 4 injection probes). Each answer saw up to 5 passages with a 35-word maximum overlap. Dense encoder BAAI/bge-small-en-v1.5 via FastEmbed 0.8.1 on Python 3.12.14.

| Mode | Words | Chunks | Threshold | Hit@1 | Hit@3 | Hit@5 | MRR | Cited fact | Refusal | Lexical support | Canary pass | Exposed probes | Median ms |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| bm25 | 200 | 60 | 0.5 | 96.4% | 100.0% | 100.0% | 0.982 | 75.0% | 50.0% | 100.0% | 100.0% | 4 | 1.60 |
| bm25 | 200 | 60 | 0.75 | 96.4% | 100.0% | 100.0% | 0.982 | 39.3% | 75.0% | 100.0% | 100.0% | 4 | 0.88 |
| dense | 200 | 60 | 0.5 | 100.0% | 100.0% | 100.0% | 1.000 | 78.6% | 62.5% | 100.0% | 100.0% | 4 | 13.93 |
| dense | 200 | 60 | 0.75 | 100.0% | 100.0% | 100.0% | 1.000 | 39.3% | 75.0% | 100.0% | 100.0% | 4 | 13.57 |
| hybrid | 200 | 60 | 0.5 | 100.0% | 100.0% | 100.0% | 1.000 | 78.6% | 62.5% | 100.0% | 100.0% | 4 | 13.52 |
| hybrid | 200 | 60 | 0.75 | 100.0% | 100.0% | 100.0% | 1.000 | 39.3% | 75.0% | 100.0% | 100.0% | 4 | 13.41 |
| bm25 | 400 | 30 | 0.5 | 96.4% | 100.0% | 100.0% | 0.982 | 75.0% | 50.0% | 100.0% | 100.0% | 4 | 2.13 |
| bm25 | 400 | 30 | 0.75 | 96.4% | 100.0% | 100.0% | 0.982 | 39.3% | 75.0% | 100.0% | 100.0% | 4 | 1.16 |
| dense | 400 | 30 | 0.5 | 100.0% | 100.0% | 100.0% | 1.000 | 78.6% | 62.5% | 100.0% | 100.0% | 4 | 14.21 |
| dense | 400 | 30 | 0.75 | 100.0% | 100.0% | 100.0% | 1.000 | 39.3% | 75.0% | 100.0% | 100.0% | 4 | 13.07 |
| hybrid | 400 | 30 | 0.5 | 100.0% | 100.0% | 100.0% | 1.000 | 78.6% | 62.5% | 100.0% | 100.0% | 4 | 13.42 |
| hybrid | 400 | 30 | 0.75 | 100.0% | 100.0% | 100.0% | 1.000 | 39.3% | 75.0% | 100.0% | 100.0% | 4 | 13.04 |

Lexical support uses a 60% content-word cutoff per answer sentence.

At 200 words and threshold 0.5, MRR was bm25 0.982, dense 1.000, hybrid 1.000, and hybrid MRR was 1.000 at 400 words. Raising the hybrid threshold to 0.75 at 200 words moved refusal accuracy from 62.5% to 75.0% and cited key-fact recovery from 78.6% to 39.3%. The euro-denominated card-limit question (u04) received the dollar-denominated limit because the top sentence's lexical score of 0.80 cleared the gate despite the unmatched currency.
