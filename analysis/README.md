# Numbered analysis workflow

The numbered scripts preserve the complete sequence from monthly observations to event catalogues and statistical summaries. Each scientific decision is kept in an ordinary Python program that can be inspected, rerun and tested.

The default processed-data location is `data/river-drought-boundaries-data-v1.0.0/data/analysis_outputs`. Override it with `RDB_DATA_ROOT` or `RDB_OUTPUTS`. A reconstruction from upstream observations also requires `RDB_PROJECT_ROOT`; see `docs/data_sources.md`.

For an analysis based on the versioned companion dataset:

```bash
export RDB_DATA_ROOT=/path/to/river-drought-boundaries-data-v1.0.0
export RDB_OUTPUTS=/path/to/new/analysis_outputs

python 02_run_audited_state_framework.py
python 03_recovery_and_sensitivity_audit.py
python 04_transient_response_audit.py
python 05_nldas_timing_audit.py
python 06_clustered_group_summaries.py
python 07_boundary_change_audit.py
```

Script 01 rebuilds monthly inputs and standardized indices from upstream daily observations. Scripts 02–06 create the primary catalogues and summaries. Script 07 extends the catalogue to test whether boundary-crossing frequencies are stable across earlier and later periods.

Write outputs to a new directory. Versioned data releases are immutable inputs and should not be overwritten.
