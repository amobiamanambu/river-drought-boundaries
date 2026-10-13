# Worked example: USGS GAGE_ID 01011000

This example reconstructs four meteorological-drought boundary classifications for USGS streamgage `01011000` (Allagash River near Allagash, Maine). The data are a small, real subset of the versioned research dataset, not simulated values.

## Files

- `data/monthly_indices_sample.csv`: monthly fixed-calibration SPEI and SSI at 3- and 6-month accumulation scales, January 2016 through December 2023.
- `data/basin_attributes_sample.csv`: selected public GAGES-II attributes for the basin.
- `data/expected_events_sample.csv`: four expected event-boundary classifications from the released catalogue.
- `create_sample_data.py`: extracts all three files from the companion dataset.
- `run_sample.py`: detects events, identifies river drought already active at meteorological onset, constructs admissible onset associations, applies maximum-cardinality one-to-one matching and classifies river state at meteorological termination.

## Run

From the repository root:

```bash
python -m pip install -e .
python examples/run_sample.py --check
```

The command writes `examples/output/reconstructed_events.csv` and exits with an error if any classification differs from `expected_events_sample.csv`.

To regenerate the included sample from the companion dataset:

```bash
python examples/create_sample_data.py /path/to/river-drought-boundaries-data-v1.0.0
```

`GAGE_ID` is stored as text so its leading zero is preserved. The monthly values derive from public USGS NWIS and gridMET inputs. Basin attributes derive from GAGES-II. The original providers remain the authoritative sources and their terms apply.
