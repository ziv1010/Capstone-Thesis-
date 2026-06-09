# Visualizer Reading Guide

Use the viewer as a paper-facing explanation browser.

## Main Interpretation

The primary explanation score is **importance**:

```text
importance = |baseline predicted-class probability - masked predicted-class probability|
```

For a case-level row, **top importance** is the largest such value among the evidence groups tested for that case.

For aggregate tables, **total importance** sums these absolute deltas over all matching groups. Use it to rank path families, relation types, and evidence types across the test set.

## Case Detail

Open a case row and read the top explanations as:

```text
When this evidence group was removed, the frozen HGT prediction changed by this much.
```

- **Signed delta > 0**: removing the group reduced confidence in the original prediction. The group supported the model's prediction.
- **Signed delta < 0**: removing the group increased confidence in the original prediction. The group was pushing against the model's prediction.
- **Flip**: masking at least one group changed the predicted class.
- **Path**: the typed legal route, such as `case->has_arguments->arguments` or `case->decided_by_bench->judge`.
- Click an explanation row to open the **Evidence Inspector**. It shows the raw graph node id, support distribution, relation notes, and other test cases where the same evidence appeared in top explanations.

## What `rev_` Means

`rev_` means the graph contains a reverse edge added for HGT message passing.

Example:

```text
arguments -> cites_precedent -> precedent
precedent -> rev_cites_precedent -> arguments
```

So `rev_cites_precedent` is not a new legal citation type. It is the reverse direction of `cites_precedent`, letting information from the precedent node flow back to the argument node. If a row says `precedent->rev_cites_precedent->arguments`, read it as: "the model was sensitive to messages travelling from this precedent back into the case argument."

## Evidence Support

The Evidence tab answers:

```text
For an important statute/provision/precedent/etc., what does its connected training neighbourhood look like?
```

- **Train support**: number of connected training cases.
- **Label -1 rate / Label 1 rate**: distribution of labels among those connected training cases.
- High support plus skewed label rate gives you the legal-neighbourhood narrative for an evidence node.
- Click an evidence row to see the exact `evidence_id`, graph index, source-search link, and case-level explanation rows where it mattered.

## Attention

Attention is shown only as a diagnostic:

- **Attention agreement**: top-k overlap between attention ranking and counterfactual ranking.
- Treat counterfactual importance as the faithful score.
- Use attention agreement only to identify explanations where both mechanisms agree.

## Validation Curves

The Validation tab compares three rankings:

- **Counterfactual**: groups sorted by absolute probability delta from the main method.
- **Attention**: groups sorted by HGT attention score.
- **Random**: shuffled groups, used as the baseline.

**Sufficiency** keeps only the top-k groups and masks all other local evidence edges. Higher preserved probability at small k is better.

**Comprehensiveness** removes only the top-k groups. Higher probability drop at small k is better.

The AUC table is the paper-level quantitative comparison. If the counterfactual ranking beats attention and random on both AUCs, it supports the claim that the counterfactual ranking is the more faithful explanation signal.

## Prediction Buckets

The bucket breakdown separates cases into:

- **high_confidence_correct**: confidence at or above the threshold and prediction equals target.
- **high_confidence_wrong**: confidence at or above the threshold and prediction does not equal target.
- **low_confidence**: confidence below the threshold.

**Evidence purity** is the sharpness of the connected training label distribution for the top evidence node:

```text
purity = max(label -1 rate, label 1 rate)
```

Use this table to check whether wrong high-confidence predictions are driven by unusually pure but misleading training neighbourhoods, larger top deltas, or different evidence types.

## Leakage Audit

The leakage audit aggregates name-like evidence types:

```text
judge, court, petitioner, respondent, lawyer
```

If these have low mean importance and low flip rate, they support the story that names/judges are not driving the model.

## Identity Shortcut Audit

The identity shortcut tab is stricter than counterfactual sensitivity. It asks:

```text
Can identity names alone predict held-out labels from train-label priors?
```

Read the table as:

- **Identity AUC**: predictive power of that identity type alone.
- **Known eval cases**: share of eval cases with at least one identity also seen in train.
- **Log-loss Δ vs domain**: negative is better than a domain-only baseline.
- **CF flip rate**: how often masking that identity type changes the frozen model prediction.

High identity-only AUC plus non-trivial counterfactual flip rate is shortcut risk. High flip rate with low identity-only AUC is weaker evidence of leakage and usually means sparse or context-specific reliance.
