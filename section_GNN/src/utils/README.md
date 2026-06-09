# src/utils

This package contains shared infrastructure used by almost every entry point.

## Files

- `io.py`: JSON/YAML helpers, directory creation, deep-merge helpers, and
  portable config path resolution.
- `logging_utils.py`: file and console logger setup.
- `pipeline.py`: shared graph-building pipeline utilities.
- `seed.py`: deterministic seed setup.
- `text_encoder.py`: sentence-transformer, Hugging Face, and hashing encoder
  wrappers.

## Path Resolution

Use `load_yaml` instead of raw `yaml.safe_load` for project configs. It expands
relative config paths against `section_GNN`, while `dump_yaml` and `dump_json`
write repository-local paths back out where possible.

This is what keeps configs reproducible across machines.
