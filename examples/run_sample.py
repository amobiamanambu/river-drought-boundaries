#!/usr/bin/env python3
"""Reconstruct event-boundary classifications for the included basin sample."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from river_drought_boundaries.events import (
    EventDefinition,
    admissible_edges,
    containing_event,
    detect_events,
    maximum_one_to_one,
)


DATA = Path(__file__).with_name("data")
OUTPUT = Path(__file__).with_name("output")
MATCHING_WINDOW_MONTHS = 6


def classify_scale(monthly: pd.DataFrame, expected: pd.DataFrame, scale: int) -> list[dict[str, object]]:
    dates = monthly.date.to_numpy()
    meteorological = detect_events(monthly[f"SPEI_{scale}_fixed"].to_numpy(), dates)
    river = detect_events(monthly[f"SSI_{scale}_fixed"].to_numpy(), dates)

    target_onsets = set(expected.loc[expected.timescale.eq(scale), "met_onset"])
    target_indices = [
        index
        for index, event in enumerate(meteorological)
        if pd.Timestamp(event["onset"]) in target_onsets
    ]
    preexisting: dict[int, bool] = {}
    initially_at_risk: list[int] = []
    for index in target_indices:
        event = meteorological[index]
        river_index = containing_event(river, int(event["onset_idx"]))
        is_preexisting = (
            river_index is not None
            and int(river[river_index]["onset_idx"]) < int(event["onset_idx"])
        )
        preexisting[index] = is_preexisting
        if not is_preexisting:
            initially_at_risk.append(index)

    edges = admissible_edges(
        meteorological,
        river,
        initially_at_risk,
        MATCHING_WINDOW_MONTHS,
    )
    assignments = {
        pair["met_index"]: pair["hyd_index"]
        for pair in maximum_one_to_one(initially_at_risk, len(river), edges)
    }

    rows: list[dict[str, object]] = []
    for index in target_indices:
        event = meteorological[index]
        assigned_index = assignments.get(index)
        assigned = None if assigned_index is None else river[assigned_index]
        met_onset_index = int(event["onset_idx"])
        met_termination_index = int(event["term_idx"])

        if preexisting[index]:
            pathway = "river_already_in_drought"
        elif assigned is None:
            pathway = "no_new_river_onset_within_window"
        elif int(assigned["onset_idx"]) - met_onset_index <= 2:
            pathway = "new_onset_by_confirmation"
        else:
            pathway = "new_onset_after_confirmation"

        active_index = containing_event(river, met_termination_index)
        active_at_termination = active_index is not None
        if active_at_termination:
            termination_state = "river_active_at_met_termination"
        elif assigned is not None and int(assigned["onset_idx"]) > met_termination_index:
            termination_state = "river_onset_pending_at_met_termination"
        elif assigned is not None:
            termination_state = "matched_river_already_recovered"
        else:
            termination_state = "no_matched_river_drought"

        rows.append(
            {
                "GAGE_ID": str(monthly.GAGE_ID.iloc[0]).zfill(8),
                "timescale": scale,
                "met_onset": pd.Timestamp(event["onset"]),
                "met_termination": pd.Timestamp(event["termination"]),
                "river_preexisting_at_met_onset": preexisting[index],
                "assigned_hyd_onset": (
                    pd.NaT if assigned is None else pd.Timestamp(assigned["onset"])
                ),
                "onset_pathway": pathway,
                "termination_state": termination_state,
                "river_active_at_met_termination": active_at_termination,
            }
        )
    return rows


def normalize(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame["GAGE_ID"] = frame.GAGE_ID.astype(str).str.zfill(8)
    for column in ("met_onset", "met_termination", "assigned_hyd_onset"):
        frame[column] = pd.to_datetime(frame[column])
    return frame.sort_values(["timescale", "met_onset"]).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Compare output with expected events")
    args = parser.parse_args()

    monthly = pd.read_csv(DATA / "monthly_indices_sample.csv", dtype={"GAGE_ID": str})
    monthly["GAGE_ID"] = monthly.GAGE_ID.str.zfill(8)
    monthly["date"] = pd.to_datetime(monthly["date"])
    expected = normalize(
        pd.read_csv(DATA / "expected_events_sample.csv", dtype={"GAGE_ID": str})
    )

    reconstructed = normalize(
        pd.DataFrame(
            classify_scale(monthly, expected, 3)
            + classify_scale(monthly, expected, 6)
        )
    )
    OUTPUT.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT / "reconstructed_events.csv"
    reconstructed.to_csv(output_path, index=False)

    if args.check:
        columns = expected.columns.tolist()
        pd.testing.assert_frame_equal(
            reconstructed[columns],
            expected[columns],
            check_dtype=False,
            check_like=False,
        )
        print(f"PASS: reconstructed {len(reconstructed)} event classifications")
    else:
        print(reconstructed.to_string(index=False))
        print(f"\nWrote {output_path}")


if __name__ == "__main__":
    main()
