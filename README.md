# River-drought event-boundary analysis

Reproducible Python code for following meteorological droughts across river-state boundaries. The workflow detects monthly drought events, distinguishes river drought already active at meteorological onset from new river-drought onset, assigns eligible events one-to-one, and follows post-termination outcomes without restricting the denominator to matched pairs.

This repository contains scientific analysis code only. It does not contain manuscript files, paper figures, figure-production code or publication-preparation utilities.

## Questions the workflow addresses

1. How often does meteorological drought begin or end while river drought is already active?
2. What is omitted when inference is restricted to matched meteorological–river event pairs?
3. Among basins initially at risk, how often is a new river-drought onset confirmed, delayed or absent within a defined association window?
4. After meteorological drought ends, which occurs first: sustained river recovery, temporary recovery, renewed meteorological forcing or continued river drought?
5. How stable are these frequencies across definitions, regions, basin classes and time periods?

## Quick start

Python 3.11 or newer is required.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
python -m unittest discover -s tests -v
python examples/run_sample.py --check
```

The worked example uses real monthly indices and attributes for USGS GAGE_ID `01011000`. It reconstructs four expected event-boundary classifications and fails loudly if any result changes. See [`examples/README.md`](examples/README.md).

## Full data workflow

Download and extract the companion research dataset, then validate its checksums, keys, study-population coverage and scientific invariants:

```bash
python -m river_drought_boundaries.validate /path/to/river-drought-boundaries-data-v1.0.0
```

The numbered programs in `analysis/` implement the complete analytical sequence:

1. Reconstruct monthly inputs and fixed-calibration standardized indices.
2. Detect meteorological and river droughts and build boundary-state catalogues.
3. Calculate recovery outcomes and sensitivity analyses.
4. Quantify one- and two-month SSI threshold excursions among no-onset events.
5. Test the incremental held-out information in NLDAS root-zone soil moisture.
6. Calculate basin-cluster regional, reference-class and hydroclimatic summaries.
7. Evaluate temporal stability in boundary-crossing frequencies.

Set `RDB_DATA_ROOT` to the extracted dataset directory. Set `RDB_OUTPUTS` to a separate working directory before rerunning analysis so the archived data remain unchanged. A reconstruction from raw public observations also requires `RDB_PROJECT_ROOT`; the expected layout is documented in [`docs/data_sources.md`](docs/data_sources.md).

```bash
export RDB_DATA_ROOT=/path/to/river-drought-boundaries-data-v1.0.0
export RDB_OUTPUTS=/path/to/new/analysis_outputs

python analysis/02_run_audited_state_framework.py
python analysis/03_recovery_and_sensitivity_audit.py
python analysis/04_transient_response_audit.py
python analysis/05_nldas_timing_audit.py
python analysis/06_clustered_group_summaries.py
python analysis/07_boundary_change_audit.py
```

## Core definitions

- Drought onset is the first month of at least three consecutive months at or below −1.0.
- Drought termination is the final drought month before two consecutive months above −0.5.
- The primary accumulation scales are 3 and 6 months.
- Fixed distributions are calibrated over 1980–2010.
- Meteorological events beginning in 2011–2024 form the primary evaluation population.
- A river drought already active at meteorological onset is classified directly and is not treated as a new onset.
- New-onset association uses maximum-cardinality one-to-one assignment within the six-month post-termination window, with overlap and lag as deterministic tie-breakers.
- River state at meteorological termination is read directly from the complete river-event catalogue, independently of assigned-pair status.
- Basin-cluster resampling preserves dependence among events from the same streamgage.

## Repository structure

- `src/river_drought_boundaries/`: reusable event detection, association and validation functions.
- `analysis/`: ordered scientific workflow.
- `examples/`: real-data worked example with a public GAGE_ID.
- `tests/`: deterministic tests for threshold persistence, missing-data censoring, admissible matching and one-to-one assignment.
- `docs/`: methods, source-data and release documentation.

Large analysis tables are distributed separately through Zenodo rather than committed to Git. Code is licensed under the MIT License. The included derived sample data are released under CC BY 4.0; upstream data retain the terms of their original providers.

## Citation

Citation metadata are provided in `CITATION.cff`. Cite the versioned software release and companion dataset used in an analysis.
