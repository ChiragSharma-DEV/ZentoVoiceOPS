# VoiceOps 2026: Step 1 vs Step 2 Evaluation Results

| Metric | Step 1 (n=52) | Step 2 all (n=52) | Step 2 holdout (n=26) |
| :--- | :--- | :--- | :--- |
| decision_correct | 15/52 (28.85%) | 44/52 (84.62%) | 23/26 (88.46%) |
| must_include_pass | 8/52 (15.38%) | 42/52 (80.77%) | 23/26 (88.46%) |
| must_not_include_violations | 0 | 0 | 0 |
| unparsed | 0/52 | 0/52 | 0/26 |
| truncated | 0/52 | 0/52 | 0/26 |
| recall@5 (answerable) | N/A | 55.67% (n=47) | 52.08% (n=24) |
| recall@10 (answerable) | N/A | 67.02% (n=47) | 64.58% (n=24) |
| hard_negative_hit_rate@5 | N/A | 0.0% (n=5) | 0.0% (n=4) |

## Step 2 ASR (Speech-to-Text Text Input) Results

| Metric | Step 2 ASR (n=52) |
| :--- | :--- |
| decision_correct | 43/52 (82.69%) |
| must_include_pass | 40/52 (76.92%) |
| must_not_include_violations | 0 |
| unparsed | 0/52 |
| truncated | 0/52 |
| recall@5 | 51.06% (n=47) |
| recall@10 | 63.83% (n=47) |
| hard_negative_hit_rate@5 | 0.0% (n=5) |
| sku_retrieval_rate | 61.54% |
