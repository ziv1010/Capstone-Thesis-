# Run Logs

Historical loose logs are collected here so generated data folders stay focused
on data artifacts.

- `cross_validated_outputs/`: logs that were previously at the top level of
  `cross_validated_outputs/`.
- `final_outputs/`: run logs that were previously mixed into individual
  `final_outputs/*_extract/` folders.

New pipeline runs may still create logs inside their configured output folder;
move long-term logs here when archiving a run.

These logs are not required to rerun the pipeline. They are useful only for
debugging or documenting previous runs.
