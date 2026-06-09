# src/preprocessing

This package contains reusable preprocessing utilities for turning extracted
case JSON data into leakage-safe model inputs.

## Files

- `extract.py`: text/entity extraction helpers for case payloads.
- `leakage.py`: conservative leakage detection and masking utilities.
- `loader.py`: typed loading and conversion helpers for cleaned cases.
- `normalize.py`: canonicalization helpers for entity names and labels.

## What Preprocessing Produces

Downstream graph builders expect each cleaned case to contain:

- a stable `case_id`
- source file metadata
- raw and mapped outcome labels
- text sections such as `preamble`, `facts`, `arguments`,
  `petitioner_arguments`, `respondent_arguments`, and `other_lawyer_arguments`
- normalized entities grouped by semantic type
- leakage audit metadata

The fixed-open entry point is
`experiments/fixed_open_pipeline/preprocess_fixed_open.py`, which writes this
schema to `data/.../processed/cleaned_cases/`.

## Leakage Policy

The preprocessing stage drops or masks outcome-bearing material before graph
construction. Typical dropped roles include `ANALYSIS`, `ISSUE`, `RATIO`,
`RLC`, and `RPC`; the exact role policy is controlled by the YAML config.
