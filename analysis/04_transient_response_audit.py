#!/usr/bin/env python3
"""Quantify short SSI threshold responses in the no-onset population.

The script reconstructs the primary three-month river-event matching before
measuring one- and two-month threshold excursions among initially at-risk
meteorological droughts with no admissible new river onset.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from _paths import OUTPUTS


CATALOGUE_PATH = OUTPUTS / "revised_state_catalogue_primary.parquet"
SERIES_PATH = OUTPUTS / "audited_indices.parquet"

ONSET_THRESHOLD = -1.0
PRIMARY_ONSET_RUN = 3
COUNTERFACTUAL_ONSET_RUN = 2
RECOVERY_THRESHOLD = -0.5
RECOVERY_RUN = 2
MATCHING_WINDOW = 6
BOOTSTRAPS = 500
SEED = 7525


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def detect_events(
    values: np.ndarray,
    dates: np.ndarray,
    onset_run: int,
    onset_threshold: float = ONSET_THRESHOLD,
    termination_threshold: float = RECOVERY_THRESHOLD,
    termination_run: int = RECOVERY_RUN,
) -> list[dict[str, object]]:
    """Audited threshold-event detector; missing months split event sequences."""
    events: list[dict[str, object]] = []
    n = len(values)
    index = 0
    while index < n:
        onset_slice = values[index : index + onset_run]
        if (
            len(onset_slice) == onset_run
            and np.isfinite(onset_slice).all()
            and (onset_slice <= onset_threshold).all()
        ):
            onset = index
            cursor = index + onset_run
            censored_by_missing = False
            while cursor < n:
                if not np.isfinite(values[cursor]):
                    censored_by_missing = True
                    break
                recovery_slice = values[cursor : cursor + termination_run]
                if (
                    len(recovery_slice) == termination_run
                    and np.isfinite(recovery_slice).all()
                    and (recovery_slice > termination_threshold).all()
                ):
                    break
                cursor += 1
            right_censored = cursor >= n or censored_by_missing
            termination = cursor - 1
            events.append(
                {
                    "onset_idx": onset,
                    "term_idx": termination,
                    "onset": pd.Timestamp(dates[onset]),
                    "termination": pd.Timestamp(dates[termination]),
                    "right_censored": right_censored,
                }
            )
            index = termination + 1
            continue
        index += 1
    return events


def containing_event(events: list[dict[str, object]], month_index: int) -> int | None:
    if not events:
        return None
    onsets = np.fromiter((int(event["onset_idx"]) for event in events), dtype=int)
    position = int(np.searchsorted(onsets, month_index, side="right") - 1)
    if position >= 0 and month_index <= int(events[position]["term_idx"]):
        return position
    return None


def reconstruct_matching(
    records: list[object], hyd_events: list[dict[str, object]]
) -> dict[str, dict[int, object]]:
    """Apply the primary pre-existing-state, edge and assignment rules."""
    preexisting: dict[int, bool] = {}
    admissible_counts: dict[int, int] = {row.row_id: 0 for row in records}
    edges: list[dict[str, int]] = []
    at_risk_ids: list[int] = []

    for row in records:
        met_onset = int(row.met_onset_idx)
        containing = containing_event(hyd_events, met_onset)
        already_active = (
            containing is not None
            and int(hyd_events[containing]["onset_idx"]) < met_onset
        )
        preexisting[row.row_id] = already_active
        if not already_active:
            at_risk_ids.append(row.row_id)

    by_id = {row.row_id: row for row in records}
    for row_id in at_risk_ids:
        row = by_id[row_id]
        met_onset = int(row.met_onset_idx)
        met_termination = int(row.met_termination_idx)
        for hyd_index, hydrologic in enumerate(hyd_events):
            hyd_onset = int(hydrologic["onset_idx"])
            lag = hyd_onset - met_onset
            if lag < 0 or hyd_onset > met_termination + MATCHING_WINDOW:
                continue
            overlap_start = max(met_onset, hyd_onset)
            overlap_end = min(met_termination, int(hydrologic["term_idx"]))
            overlap = max(0, overlap_end - overlap_start + 1)
            edges.append(
                {
                    "row_id": row_id,
                    "hyd_index": hyd_index,
                    "lag": lag,
                    "overlap": overlap,
                }
            )
            admissible_counts[row_id] += 1

    assignments: dict[int, int] = {}
    if at_risk_ids and hyd_events and edges:
        row_lookup = {row_id: index for index, row_id in enumerate(at_risk_ids)}
        score = np.zeros((len(at_risk_ids), len(hyd_events)), dtype=float)
        lookup: dict[tuple[int, int], dict[str, int]] = {}
        for edge in edges:
            row_index = row_lookup[edge["row_id"]]
            hyd_index = edge["hyd_index"]
            score[row_index, hyd_index] = (
                1_000_000_000 + 1_000 * edge["overlap"] - edge["lag"]
            )
            lookup[(row_index, hyd_index)] = edge
        rows, columns = linear_sum_assignment(score, maximize=True)
        for row_index, hyd_index in zip(rows, columns):
            if (row_index, hyd_index) in lookup:
                assignments[at_risk_ids[row_index]] = hyd_index

    return {
        "preexisting": preexisting,
        "admissible_counts": admissible_counts,
        "assignments": assignments,
    }


def max_true_run(mask: np.ndarray) -> int:
    longest = 0
    current = 0
    for value in mask:
        if bool(value):
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return longest


def two_month_status(
    row_id: int,
    reconstruction: dict[str, dict[int, object]],
) -> str:
    if bool(reconstruction["preexisting"][row_id]):
        return "River already active"
    n_admissible = int(reconstruction["admissible_counts"][row_id])
    if row_id in reconstruction["assignments"]:
        return "Assigned new onset"
    if n_admissible > 0:
        return "Assignment competition"
    return "No two-month onset"


def cluster_interval(
    frame: pd.DataFrame,
    outcome: str,
    rng: np.random.Generator,
) -> tuple[float, float, float]:
    grouped = frame.groupby("GAGE_ID")[outcome].agg(["sum", "count"])
    numerators = grouped["sum"].to_numpy(dtype=float)
    denominators = grouped["count"].to_numpy(dtype=float)
    estimate = float(numerators.sum() / denominators.sum())
    draws = np.empty(BOOTSTRAPS, dtype=float)
    for index in range(BOOTSTRAPS):
        selected = rng.integers(0, len(grouped), size=len(grouped))
        draws[index] = numerators[selected].sum() / denominators[selected].sum()
    lower, upper = np.quantile(draws, [0.025, 0.975])
    return estimate, float(lower), float(upper)


def main() -> None:
    OUTPUTS.mkdir(parents=True, exist_ok=True)

    catalogue = pd.read_parquet(CATALOGUE_PATH).copy()
    catalogue["GAGE_ID"] = catalogue.GAGE_ID.astype(str).str.zfill(8)
    for column in ("met_onset", "met_termination", "assigned_hyd_onset"):
        catalogue[column] = pd.to_datetime(catalogue[column])
    catalogue = catalogue.sort_values(["GAGE_ID", "timescale", "met_onset"]).reset_index(drop=True)

    series = pd.read_parquet(
        SERIES_PATH,
        columns=["GAGE_ID", "date", "SSI_3_fixed", "SSI_6_fixed", "primary_population"],
    )
    series = series[series.primary_population].copy()
    series["GAGE_ID"] = series.GAGE_ID.astype(str).str.zfill(8)
    series["date"] = pd.to_datetime(series.date)
    series = series.sort_values(["GAGE_ID", "date"]).reset_index(drop=True)

    catalogue_by_gage = {
        gage: group.copy() for gage, group in catalogue.groupby("GAGE_ID", sort=False)
    }
    diagnostic_rows: list[dict[str, object]] = []
    reconstruction_rows: list[dict[str, object]] = []

    for gage, monthly in series.groupby("GAGE_ID", sort=False):
        if gage not in catalogue_by_gage:
            continue
        monthly = monthly.sort_values("date").reset_index(drop=True)
        dates = monthly.date.to_numpy()
        gage_catalogue = catalogue_by_gage[gage]
        for timescale in (3, 6):
            scale_catalogue = gage_catalogue.loc[
                gage_catalogue.timescale.eq(timescale)
            ].copy()
            if scale_catalogue.empty:
                continue
            scale_catalogue = scale_catalogue.sort_values("met_onset").reset_index(drop=True)
            scale_catalogue["row_id"] = np.arange(len(scale_catalogue), dtype=int)
            records = list(scale_catalogue.itertuples(index=False))
            values = monthly[f"SSI_{timescale}_fixed"].to_numpy(dtype=float)

            for row in records:
                onset_index = int(row.met_onset_idx)
                termination_index = int(row.met_termination_idx)
                if pd.Timestamp(dates[onset_index]) != pd.Timestamp(row.met_onset):
                    raise RuntimeError(f"Meteorological onset index mismatch: {gage}, {timescale}")
                if pd.Timestamp(dates[termination_index]) != pd.Timestamp(row.met_termination):
                    raise RuntimeError(f"Meteorological termination index mismatch: {gage}, {timescale}")

            hyd_events_primary = detect_events(values, dates, PRIMARY_ONSET_RUN)
            primary_reconstruction = reconstruct_matching(records, hyd_events_primary)
            hyd_events_two_month = detect_events(values, dates, COUNTERFACTUAL_ONSET_RUN)
            two_month_reconstruction = reconstruct_matching(records, hyd_events_two_month)

            for row in records:
                row_id = int(row.row_id)
                reconstructed_preexisting = bool(primary_reconstruction["preexisting"][row_id])
                reconstructed_n_admissible = int(
                    primary_reconstruction["admissible_counts"][row_id]
                )
                assigned_index = primary_reconstruction["assignments"].get(row_id)
                reconstructed_assigned = (
                    pd.NaT
                    if assigned_index is None
                    else pd.Timestamp(hyd_events_primary[assigned_index]["onset"])
                )
                primary_assigned = pd.Timestamp(row.assigned_hyd_onset) if pd.notna(row.assigned_hyd_onset) else pd.NaT
                assignment_matches = (
                    (pd.isna(reconstructed_assigned) and pd.isna(primary_assigned))
                    or reconstructed_assigned == primary_assigned
                )
                reconstruction_rows.append(
                    {
                        "GAGE_ID": gage,
                        "timescale": timescale,
                        "met_onset": row.met_onset,
                        "preexisting_matches": reconstructed_preexisting
                        == bool(row.river_preexisting_at_met_onset),
                        "admissible_count_matches": reconstructed_n_admissible
                        == int(row.n_admissible_hydrologic_onsets),
                        "assignment_matches": assignment_matches,
                    }
                )

                is_primary_no_admissible = (
                    not bool(row.river_preexisting_at_met_onset)
                    and int(row.n_admissible_hydrologic_onsets) == 0
                )
                if not is_primary_no_admissible:
                    continue

                window_start = int(row.met_onset_idx)
                window_end = int(row.met_termination_idx) + MATCHING_WINDOW
                window_values = values[window_start : window_end + 1]
                if len(window_values) != window_end - window_start + 1:
                    raise RuntimeError(f"Incomplete matching window: {gage}, {timescale}")
                if not np.isfinite(window_values).all():
                    raise RuntimeError(f"Non-finite SSI in eligible matching window: {gage}, {timescale}")
                below = window_values <= ONSET_THRESHOLD
                maximum_run = max_true_run(below)
                if maximum_run >= PRIMARY_ONSET_RUN:
                    raise RuntimeError(
                        f"Three-month SSI run inside primary no-admissible cohort: {gage}, {timescale}, {row.met_onset}"
                    )
                threshold_indices = np.flatnonzero(below)
                first_index = (
                    np.nan if len(threshold_indices) == 0 else int(threshold_indices[0])
                )
                if len(threshold_indices) == 0:
                    phase = "No threshold crossing"
                else:
                    absolute_first = window_start + int(first_index)
                    if absolute_first <= window_start + 2:
                        phase = "By formal confirmation"
                    elif absolute_first <= int(row.met_termination_idx):
                        phase = "Later during meteorological drought"
                    else:
                        phase = "Post-termination window"

                diagnostic_rows.append(
                    {
                        "GAGE_ID": gage,
                        "timescale": timescale,
                        "met_onset": row.met_onset,
                        "met_termination": row.met_termination,
                        "window_end": pd.Timestamp(dates[window_end]),
                        "months_at_or_below_minus_one": int(below.sum()),
                        "maximum_consecutive_months_at_or_below_minus_one": maximum_run,
                        "minimum_ssi_in_window": float(np.min(window_values)),
                        "transient_response": maximum_run in (1, 2),
                        "first_crossing_months_from_met_onset": first_index,
                        "first_crossing_phase": phase,
                        "two_month_status": two_month_status(
                            row_id, two_month_reconstruction
                        ),
                    }
                )

    reconstruction = pd.DataFrame(reconstruction_rows)
    audit = {
        "n_audited_events": int(len(reconstruction)),
        "preexisting_mismatches": int((~reconstruction.preexisting_matches).sum()),
        "admissible_count_mismatches": int((~reconstruction.admissible_count_matches).sum()),
        "assignment_mismatches": int((~reconstruction.assignment_matches).sum()),
    }
    if any(audit[key] for key in audit if key != "n_audited_events"):
        raise RuntimeError(f"Frozen catalogue reconstruction failed: {audit}")

    events = pd.DataFrame(diagnostic_rows)
    events.to_csv(OUTPUTS / "transient_river_response_event_catalogue.csv", index=False)
    events.to_parquet(
        OUTPUTS / "transient_river_response_event_catalogue.parquet", index=False
    )

    summary_rows: list[dict[str, object]] = []
    phase_rows: list[dict[str, object]] = []
    two_month_rows: list[dict[str, object]] = []
    interval_rows: list[dict[str, object]] = []
    rng = np.random.default_rng(SEED)
    for timescale, group in events.groupby("timescale"):
        primary_scale = catalogue.loc[catalogue.timescale.eq(timescale)]
        initially_at_risk = primary_scale.loc[~primary_scale.river_preexisting_at_met_onset]
        counts = group.maximum_consecutive_months_at_or_below_minus_one.value_counts()
        transient_count = int(group.transient_response.sum())
        summary_rows.append(
            {
                "timescale": int(timescale),
                "n_initially_at_risk": int(len(initially_at_risk)),
                "n_no_admissible_new_onset": int(len(group)),
                "pct_no_admissible_among_initially_at_risk": 100
                * len(group)
                / len(initially_at_risk),
                "n_no_threshold_crossing": int(counts.get(0, 0)),
                "pct_no_threshold_crossing": 100 * counts.get(0, 0) / len(group),
                "n_one_month_maximum": int(counts.get(1, 0)),
                "pct_one_month_maximum": 100 * counts.get(1, 0) / len(group),
                "n_two_month_maximum": int(counts.get(2, 0)),
                "pct_two_month_maximum": 100 * counts.get(2, 0) / len(group),
                "n_transient_response": transient_count,
                "pct_transient_response": 100 * transient_count / len(group),
                "pct_transient_among_initially_at_risk": 100
                * transient_count
                / len(initially_at_risk),
            }
        )
        transient_group = group.loc[group.transient_response]
        for phase, count in transient_group.first_crossing_phase.value_counts().items():
            phase_rows.append(
                {
                    "timescale": int(timescale),
                    "phase": phase,
                    "n_events": int(count),
                    "pct_transient_events": 100 * count / len(transient_group),
                }
            )
        for status, count in transient_group.two_month_status.value_counts().items():
            two_month_rows.append(
                {
                    "timescale": int(timescale),
                    "status": status,
                    "n_events": int(count),
                    "pct_transient_events": 100 * count / len(transient_group),
                    "pct_no_admissible_events": 100 * count / len(group),
                    "pct_initially_at_risk_events": 100
                    * count
                    / len(initially_at_risk),
                }
            )
        estimate, lower, upper = cluster_interval(group, "transient_response", rng)
        interval_rows.append(
            {
                "timescale": int(timescale),
                "metric": "transient_share",
                "estimate": estimate,
                "lower": lower,
                "upper": upper,
                "bootstrap_unit": "basin",
                "n_bootstraps": BOOTSTRAPS,
            }
        )

    summary = pd.DataFrame(summary_rows).sort_values("timescale")
    phases = pd.DataFrame(phase_rows).sort_values(["timescale", "phase"])
    two_month = pd.DataFrame(two_month_rows).sort_values(["timescale", "status"])
    intervals = pd.DataFrame(interval_rows).sort_values(["timescale", "metric"])
    summary.to_csv(OUTPUTS / "transient_river_response_summary.csv", index=False)
    phases.to_csv(OUTPUTS / "transient_first_crossing_phase_summary.csv", index=False)
    two_month.to_csv(OUTPUTS / "transient_two_month_counterfactual_summary.csv", index=False)
    intervals.to_csv(OUTPUTS / "transient_cluster_bootstrap_intervals.csv", index=False)
    (OUTPUTS / "matching_reconstruction_audit.json").write_text(
        json.dumps(audit, indent=2) + "\n"
    )

    provenance = {
        "primary_catalogue": str(CATALOGUE_PATH),
        "primary_catalogue_sha256": sha256(CATALOGUE_PATH),
        "monthly_series": str(SERIES_PATH),
        "monthly_series_sha256": sha256(SERIES_PATH),
        "primary_no_onset_denominator": (
            "Initially at-risk meteorological events with zero admissible "
            "three-month SSI onsets; one-to-one assignment conflicts are excluded."
        ),
        "transient_definition": (
            "At least one SSI month <= -1.0 between meteorological onset and six "
            "months after meteorological termination, with no run of three such months."
        ),
        "two_month_sensitivity": (
            "Meteorological origins and the association window are fixed; only the "
            "SSI onset persistence is changed from three to two months before rerunning "
            "pre-existing-state classification and one-to-one matching."
        ),
        "bootstrap_resamples": BOOTSTRAPS,
        "seed": SEED,
    }
    (OUTPUTS / "analysis_provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n"
    )

    print(json.dumps(audit, indent=2))
    print(summary.to_string(index=False))
    print("\nTwo-month sensitivity among transient responses")
    print(two_month.to_string(index=False))


if __name__ == "__main__":
    main()
