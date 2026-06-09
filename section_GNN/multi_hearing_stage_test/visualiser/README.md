# visualiser

Small visualiser for browsing multi-hearing stage-transition outputs.

## Files

- `app.py`: visualiser application.
- `run_app.sh`: shell wrapper to launch the app.

## Inputs

The app expects outputs from the multi-hearing workflow, especially:

- `multi_hearing_stage_test/outputs/stage_manifest.csv`
- `multi_hearing_stage_test/outputs/inference/predictions.csv`
- `multi_hearing_stage_test/outputs/analysis/stage_transitions.csv`
- per-case factor reports under `multi_hearing_stage_test/outputs/analysis/`

## Run

From `section_GNN`:

```bash
bash multi_hearing_stage_test/visualiser/run_app.sh
```
