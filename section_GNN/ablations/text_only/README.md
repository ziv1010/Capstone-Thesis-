# 📝 text_only — Text-Only Graph Ablation

> Part of [`ablations/`](../README.md).

Removes legal-entity and authority-node structure so the model sees only **case and
text-section information** — testing whether performance comes from text encodings alone or
from the heterogeneous legal graph structure.

## 🧬 Node Policy

Kept: `case`, `preamble`, `facts`, `arguments`, and party-specific argument sections where
configured. Removed/disabled: `statute`, `provision`, `precedent`, `judge`, `lawyer`,
`petitioner`, `respondent`.

## ▶️ Run

```bash
bash ablations/text_only/cross_bucket_total_dataset/run.sh
```

Each bucket folder has a `config.yaml`; some also have a `config_hashing.yaml` for
hashing-encoder tests.

---

⬆️ Back to [`ablations/`](../README.md)
