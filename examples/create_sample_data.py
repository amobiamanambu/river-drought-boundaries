#!/usr/bin/env python3
"""Extract the documented one-basin example from the versioned dataset."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


GAGE_ID = "01011000"
START = pd.Timestamp("2016-01-01")
END = pd.Timestamp("2023-12-01")
SERIES_COLUMNS = [
    "GAGE_ID",
    "date",
    "SPEI_3_fixed",
    "SSI_3_fixed",
    "SPEI_6_fixed",
    "SSI_6_fixed",
]
ATTRIBUTE_COLUMNS = [
    "GAGE_ID",
    "HUC02",
    "AGGECOREGION",
    "reference_class",
    "DRAIN_SQKM",
    "LAT_GAGE",
    "LNG_GAGE",
    "aridity",
    "WTDEPAVE",
    "SNOW_PCT_PRECIP",
]
EVENT_COLUMNS = [
    "GAGE_ID",
    "timescale",
    "met_onset",
    "met_termination",
    "river_preexisting_at_met_onset",
    "assigned_hyd_onset",
    "onset_pathway",
    "termination_state",
    "river_active_at_met_termination",
]


def normalized_identifier(series: pd.Series) -> pd.Series:
    return series.astype(str).str.zfill(8)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_root", type=Path, help="Extracted companion dataset")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).with_name("data"),
        help="Destination directory (default: examples/data)",
    )
    args = parser.parse_args()
    outputs = args.data_root.resolve() / "data" / "analysis_outputs"
    args.output.mkdir(parents=True, exist_ok=True)

    series = pd.read_parquet(outputs / "audited_indices.parquet", columns=SERIES_COLUMNS)
    series["GAGE_ID"] = normalized_identifier(series["GAGE_ID"])
    series["date"] = pd.to_datetime(series["date"])
    series = series.loc[
        series.GAGE_ID.eq(GAGE_ID) & series.date.between(START, END), SERIES_COLUMNS
    ].sort_values("date")
    if len(series) != 96:
        raise RuntimeError(f"Expected 96 monthly records; found {len(series)}")

    attributes = pd.read_csv(
        args.data_root / "data" / "basins" / "basin_attributes_used.csv",
        dtype={"GAGE_ID": str, "HUC02": str},
    )
    attributes["GAGE_ID"] = normalized_identifier(attributes["GAGE_ID"])
    attributes = attributes.loc[attributes.GAGE_ID.eq(GAGE_ID), ATTRIBUTE_COLUMNS]
    if len(attributes) != 1:
        raise RuntimeError(f"Expected one attribute row; found {len(attributes)}")

    events = pd.read_parquet(outputs / "revised_state_catalogue_primary.parquet")
    events["GAGE_ID"] = normalized_identifier(events["GAGE_ID"])
    events["met_onset"] = pd.to_datetime(events["met_onset"])
    events = events.loc[
        events.GAGE_ID.eq(GAGE_ID)
        & events.met_onset.between("2018-01-01", "2021-12-01"),
        EVENT_COLUMNS,
    ].sort_values(["timescale", "met_onset"])
    if len(events) != 4:
        raise RuntimeError(f"Expected four catalogue events; found {len(events)}")

    series.to_csv(args.output / "monthly_indices_sample.csv", index=False, float_format="%.8f")
    attributes.to_csv(args.output / "basin_attributes_sample.csv", index=False)
    events.to_csv(args.output / "expected_events_sample.csv", index=False)


if __name__ == "__main__":
    main()
