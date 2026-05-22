# Stage Visualiser

Interactive Dash app for inspecting how a case changes across the Fixed GPU OpenNyai pipeline stages.

## Stages

- Stage 1: NER and rhetorical-role extraction.
- Stage 2: OpenNyai summary enrichment.
- Stage 3: Mistral outcome labeling.
- Stage 4: cross-validated outcome augmentation.

## Running

```bash
bash STAGE_VISUALISER/run_app.sh 8053
```

The app expects local `Fixed_GPU_OpenNyai/final_outputs/` and `Fixed_GPU_OpenNyai/cross_validated_outputs/` directories. Those generated outputs are intentionally ignored.
