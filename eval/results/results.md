| Stage | Context precision | Context recall | Faithfulness | Citation accuracy | Species mix-up ↓ | Emergency detection | Correct decline | Latency p50 / p95 |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 0 · Baseline (dense top-k) | 0.53 | 0.69 | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ |
| 1 · + Hybrid (BM25 + dense, RRF) | 0.48 | 0.61 | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ |
| 2 · + BGE reranker | 0.58 | 0.72 | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ |
| 3 · + Species filters & safety | 0.74 | 0.87 | 0.95 | 63% | 0% | 100% | 100% | 3.4 s / 13.6 s |

**Retrieval and safety details**

| Stage | Hit@5 | MRR | Retrieval mix-up ↓ | Missed species traps ↓ | False emergency ↓ | Over-decline ↓ | Retrieval p50 / p95 |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 0 · Baseline (dense top-k) | 69% | 0.51 | 23% | ⏳ | ⏳ | ⏳ | 0.1 s / 0.1 s |
| 1 · + Hybrid (BM25 + dense, RRF) | 61% | 0.45 | 27% | ⏳ | ⏳ | ⏳ | 0.1 s / 0.1 s |
| 2 · + BGE reranker | 72% | 0.59 | 23% | ⏳ | ⏳ | ⏳ | 4.2 s / 11.3 s |
| 3 · + Species filters & safety | 87% | 0.70 | 1% | 0% | 0% | 4% | 3.0 s / 13.3 s |
