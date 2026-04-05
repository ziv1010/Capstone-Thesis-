# Case Star Graph + Global Authority Graph for Pre-Judgment Outcome Prediction

This project builds a leakage-safe heterogeneous graph neural network pipeline for legal outcome prediction before judgment.

The pipeline reads per-case JSON files produced by the OpenNyai-style extraction pipeline, removes outcome-bearing content, builds local case star graphs, merges them through shared authority/context nodes, and trains a PyTorch Geometric hetero GNN over case nodes.

## Why this is pre-judgment prediction

The model is designed to predict the appellant or petitioner outcome from information that would plausibly exist before the final order is issued. The pipeline therefore uses:

- preamble text
- facts summary
- arguments summary
- pre-judgment entities such as court, judge, statutes, provisions, lawyers, parties, precedents, and locations

It explicitly excludes final outcome text, operative orders, and decision summaries from model inputs.

## Leakage Prevention in Pre-Judgment Legal Prediction

The preprocessing layer applies hard exclusions and audits them per case.

Excluded top-level fields:

- `case_outcome_label`
- `case_outcome_score`
- `llm_case_outcome`
- `decision_text`
- `rpc_texts`
- `short_explanation`
- `raw_model_response`

Excluded summary fields:

- `raw_result.summary.decision`
- `raw_result.summary.ANALYSIS`
- `raw_result.summary.issue`

Excluded annotations:

- any annotation with `summary_section == "decision"`
- any annotation with `summary_section` in conservative blocked sections such as `ANALYSIS` and `issue`
- any annotation carrying `RPC` or `RLC`
- any annotation whose text matches an explicit outcome phrase

Regex-based leakage masking is also applied to retained text. Examples include:

- `petition stands disposed of`
- `appeal allowed`
- `appeal dismissed`
- `set aside`
- `quashed`
- `liberty granted`

Each processed case gets a dedicated audit JSON in `data/audits/` listing:

- fields dropped
- annotations dropped
- decision text removal flag
- leakage phrases matched
- retained text lengths

## Graph Schema

### Local Case Star Graph

Each case has one central `case` node connected to:

- text nodes: `preamble`, `facts`, `arguments`
- party and authority nodes: `petitioner`, `respondent`, `court`, `judge`, `lawyer`
- legal citation nodes: `statute`, `provision`, optional `precedent`
- context nodes: `org`, `gpe`, `date`, `case_number`

Key edges:

- `case -> has_preamble -> preamble`
- `case -> has_facts -> facts`
- `case -> has_arguments -> arguments`
- `case -> has_petitioner -> petitioner`
- `case -> has_respondent -> respondent`
- `case -> heard_in -> court`
- `case -> decided_by_bench -> judge`
- `case -> has_lawyer -> lawyer`
- `arguments -> cites_statute -> statute`
- `arguments -> cites_provision -> provision`
- `provision -> belongs_to_statute -> statute`
- optional mention edges for `precedent`, `org`, `gpe`, `date`, `case_number`

No decision node is created.

### Global Authority Graph

All local case star graphs are merged into one heterogeneous graph by sharing normalized nodes across cases for:

- `court`
- `judge`
- `lawyer`
- `statute`
- `provision`
- `precedent`
- `org`
- `gpe`
- `date`
- `case_number`

By default, `petitioner` and `respondent` nodes stay case-local because party resolution is noisier.

The first version deliberately does not add dense case-to-case similarity edges.

## Model

Default model:

- `HGTConv`-based hetero GNN
- 2 message-passing layers
- learned node-type offsets
- case-node MLP head for classification

Fallback:

- `HeteroConv` with relation-specific `SAGEConv`

Outputs:

- logits per case node
- accuracy
- macro F1
- micro F1
- per-class precision/recall/F1
- confusion matrix
- ROC-AUC and PR-AUC for binary runs

## 3-Way Setup

There is now a separate 3-way configuration for predicting `-1`, `0`, and `1` from:

```text
/scratch/ziv_baretto/Thesis_Ziv/Capstone-Thesis-/section_GNN/data/input/augmented_jsons
```

Config:

```text
/scratch/ziv_baretto/Thesis_Ziv/Capstone-Thesis-/section_GNN/configs/gnn_case_star_augmented_jsons_3way.yaml
```

This config keeps `postponed_or_procedural` cases instead of dropping them and maps labels as:

- `appellant_lost` → `-1`
- `postponed_or_procedural` → `0`
- `appellant_won` → `1`

It writes to separate folders so the binary runs stay untouched:

- `section_GNN/data/augmented_jsons_3way/`
- `section_GNN/outputs/augmented_jsons_3way/`

End-to-end run:

```bash
micromamba run -p /scratch/ziv_baretto/Thesis_Ziv/Capstone-Thesis-/GNN/.micromamba/gnn_case_star bash -lc '
python /scratch/ziv_baretto/Thesis_Ziv/Capstone-Thesis-/section_GNN/scripts/preprocess_cases.py \
  --config /scratch/ziv_baretto/Thesis_Ziv/Capstone-Thesis-/section_GNN/configs/gnn_case_star_augmented_jsons_3way.yaml && \
python /scratch/ziv_baretto/Thesis_Ziv/Capstone-Thesis-/section_GNN/scripts/build_graph.py \
  --config /scratch/ziv_baretto/Thesis_Ziv/Capstone-Thesis-/section_GNN/configs/gnn_case_star_augmented_jsons_3way.yaml && \
python /scratch/ziv_baretto/Thesis_Ziv/Capstone-Thesis-/section_GNN/scripts/train_gnn.py \
  --config /scratch/ziv_baretto/Thesis_Ziv/Capstone-Thesis-/section_GNN/configs/gnn_case_star_augmented_jsons_3way.yaml \
  --run-name augmented_jsons_3way
'
```

## Project Layout

```text
GNN/
  configs/
  data/
    processed/
    embeddings_cache/
    graph_cache/
    audits/
  envs/
  outputs/
  scripts/
  src/
    preprocessing/
    graph/
    models/
    training/
    utils/
```

## Environment Setup

Create the local micromamba environment:

```bash
micromamba create -y -p /scratch/ziv_baretto/Thesis_Ziv/Capstone-Thesis-/GNN/.micromamba/gnn_case_star \
  -f /scratch/ziv_baretto/Thesis_Ziv/Capstone-Thesis-/GNN/envs/gnn_case_star.yaml
```

Run commands inside it with:

```bash
micromamba run -p /scratch/ziv_baretto/Thesis_Ziv/Capstone-Thesis-/GNN/.micromamba/gnn_case_star <command>
```

## Main Scripts

### 1. Preprocess cases

```bash
micromamba run -p /scratch/ziv_baretto/Thesis_Ziv/Capstone-Thesis-/GNN/.micromamba/gnn_case_star \
  python /scratch/ziv_baretto/Thesis_Ziv/Capstone-Thesis-/GNN/scripts/preprocess_cases.py \
  --config /scratch/ziv_baretto/Thesis_Ziv/Capstone-Thesis-/GNN/configs/gnn_case_star.yaml
```

Outputs:

- `data/processed/cleaned_cases/*.json`
- `data/processed/normalized_entities/*.json`
- `data/audits/*.json`

### 2. Build the graph

```bash
micromamba run -p /scratch/ziv_baretto/Thesis_Ziv/Capstone-Thesis-/GNN/.micromamba/gnn_case_star \
  python /scratch/ziv_baretto/Thesis_Ziv/Capstone-Thesis-/GNN/scripts/build_graph.py \
  --config /scratch/ziv_baretto/Thesis_Ziv/Capstone-Thesis-/GNN/configs/gnn_case_star.yaml
```

Outputs:

- `data/graph_cache/case_star_global_graph.pt`
- `data/graph_cache/graph_metadata.json`
- `data/graph_cache/node_mappings.json`
- `data/graph_cache/relation_mappings.json`
- `data/graph_cache/split_assignments.json`
- `data/graph_cache/graph_debug_samples.json`

### 3. Train the GNN

```bash
micromamba run -p /scratch/ziv_baretto/Thesis_Ziv/Capstone-Thesis-/GNN/.micromamba/gnn_case_star \
  python /scratch/ziv_baretto/Thesis_Ziv/Capstone-Thesis-/GNN/scripts/train_gnn.py \
  --config /scratch/ziv_baretto/Thesis_Ziv/Capstone-Thesis-/GNN/configs/gnn_case_star.yaml \
  --run-name full_hgt_run
```

Outputs:

- `outputs/models/<run_name>/model.pt`
- `outputs/models/<run_name>/metrics.json`
- `outputs/models/<run_name>/predictions.csv`
- `outputs/models/<run_name>/confusion_matrix_test.png`

### 4. Run ablations

```bash
micromamba run -p /scratch/ziv_baretto/Thesis_Ziv/Capstone-Thesis-/GNN/.micromamba/gnn_case_star \
  python /scratch/ziv_baretto/Thesis_Ziv/Capstone-Thesis-/GNN/scripts/run_ablation.py \
  --config /scratch/ziv_baretto/Thesis_Ziv/Capstone-Thesis-/GNN/configs/gnn_case_star.yaml \
  --variants full_star_global without_judge without_statute_provision text_only
```

## Example Runs

### Single-case preprocessing sanity check

This is useful for inspecting leakage removal on one sample JSON before building a trainable dataset.

```bash
micromamba run -p /scratch/ziv_baretto/Thesis_Ziv/Capstone-Thesis-/GNN/.micromamba/gnn_case_star \
  python /scratch/ziv_baretto/Thesis_Ziv/Capstone-Thesis-/GNN/scripts/preprocess_cases.py \
  --config /scratch/ziv_baretto/Thesis_Ziv/Capstone-Thesis-/GNN/configs/gnn_case_star_sanity.yaml \
  --limit 1
```

Sample input file in this dataset:

`/scratch/ziv_baretto/Thesis_Ziv/Capstone-Thesis-/OpenNyai/outputs/current_output/combined_mistral24b_case_outcomes/augmented_jsons/Abhijeet_Suryakant_Maske_And_Anr_vs_The_State_Of_Maharashtra_on_1_March_2022.json`

Inspect:

- `data/processed/cleaned_cases/Abhijeet_Suryakant_Maske_And_Anr_vs_The_State_Of_Maharashtra_on_1_March_2022.json`
- `data/audits/Abhijeet_Suryakant_Maske_And_Anr_vs_The_State_Of_Maharashtra_on_1_March_2022.json`

### Fast end-to-end sanity run

This uses the hashing encoder and a short HGT schedule.

```bash
micromamba run -p /scratch/ziv_baretto/Thesis_Ziv/Capstone-Thesis-/GNN/.micromamba/gnn_case_star \
  python /scratch/ziv_baretto/Thesis_Ziv/Capstone-Thesis-/GNN/scripts/preprocess_cases.py \
  --config /scratch/ziv_baretto/Thesis_Ziv/Capstone-Thesis-/GNN/configs/gnn_case_star_sanity.yaml \
  --limit 60

micromamba run -p /scratch/ziv_baretto/Thesis_Ziv/Capstone-Thesis-/GNN/.micromamba/gnn_case_star \
  python /scratch/ziv_baretto/Thesis_Ziv/Capstone-Thesis-/GNN/scripts/build_graph.py \
  --config /scratch/ziv_baretto/Thesis_Ziv/Capstone-Thesis-/GNN/configs/gnn_case_star_sanity.yaml \
  --limit 60

micromamba run -p /scratch/ziv_baretto/Thesis_Ziv/Capstone-Thesis-/GNN/.micromamba/gnn_case_star \
  python /scratch/ziv_baretto/Thesis_Ziv/Capstone-Thesis-/GNN/scripts/train_gnn.py \
  --config /scratch/ziv_baretto/Thesis_Ziv/Capstone-Thesis-/GNN/configs/gnn_case_star_sanity.yaml \
  --run-name sanity_hgt_hashing
```

In the verification run shipped in this folder:

- 60 raw JSONs were preprocessed
- 48 non-procedural cases remained for binary training
- the graph contained cached node embeddings in `data/embeddings_cache/`
- training outputs were written to `outputs/models/sanity_hgt_hashing/`

## Training & Split Policy (Transductive Masking)

Supported physical split assignments:

- `random`
- `year`
- `court`

The architecture treats the dataset as a **transductive global graph**. This has major implications for how training and evaluation work:

### 1. The Global Forward Pass
During every epoch, the GNN doesn't look at cases one-by-one. It feeds the **entire global graph** (containing every Train, Val, and Test case) into the network simultaneously. By the end of the forward pass, the model outputs a prediction (a `logit` probability) for **every single case node** at once.

### 2. The Masking Trick
To prevent the model from cheating on the Test set, every `case` node holds a boolean array flag (`train_mask`, `val_mask`, or `test_mask`).

### 3. Calculating Loss (Training)
When calculating the Error/Loss to actually update the neural network weights, the PyTorch code mathematically "slices" the predictions.
```python
loss = cross_entropy_loss(logits[train_mask], true_labels[train_mask])
```
The GNN generated predictions for the Val and Test cases, but the optimizer completely ignores them. The model's weights are updated **exclusively based on its performance on the `train_mask` cases**.

### 4. Evaluation 
During validation or testing, the exact same slicing happens, but targeting the respective mask:
```python
val_metrics = calculate_metrics(logits[val_mask], true_labels[val_mask])
```

### The Transductive "Leak" Caveat
Because the full graph is evaluated simultaneously, and nodes like `[judge]` are global, the embeddings of the `Train` cases pass messages to the `Judge` taking their labels implicitly with them. In the *same forward pass*, that updated `Judge` passes messages into the `Test` cases. The model never explicitly peeks at the Test **labels**, but the structural mathematical context of the Train set profoundly influences the Test cases during message passing.

## Known Limitations

- Entity normalization is conservative and string-based, not a full legal entity resolver.
- Party nodes are local by default to avoid unsafe cross-case merges.
- The dataset is small and class-imbalanced, so validation and test F1 can be unstable.
- The first version does not use raw full-text sentence graphs or dense case similarity edges.
- `ANALYSIS` is conservatively excluded to avoid post-judgment reasoning leakage, which may remove some potentially useful but risky context.
