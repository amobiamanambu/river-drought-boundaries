#!/usr/bin/env python3
"""Build drought pathways around directly observed river state.

The workflow applies six ordered rules:

1. Eligible meteorological droughts that begin during an existing SSI drought
   receive their own onset class and are not allowed to claim a later SSI
   onset in the one-to-one assignment.
2. One-to-one assignment is solved over eligible, at-risk
   meteorological origins, so out-of-scope origins cannot consume a match.
3. River state at meteorological termination is read directly from the SSI
   event sequence, independently of the status of the assigned event.
4. Recovery trajectories start from every meteorological origin for which the
   river is actually in drought at meteorological termination.  A new
   meteorological drought within the horizon is retained as ``re_forced``.
5. Pair-only timing summaries, full-denominator pathway shares, static-basin
   stratification, truncation, reference-class and operational-persistence
   summaries are written as analysis tables.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import special, stats
from scipy.optimize import linear_sum_assignment, minimize
from scipy.special import betaln

from _paths import BASIN_ATTRIBUTES, OUTPUTS, PROJECT_ROOT


ROOT = PROJECT_ROOT
RES = OUTPUTS
GAGES = ROOT / "gagesii_csv"
OUT = RES
OUT.mkdir(parents=True, exist_ok=True)

EVAL_START = pd.Timestamp("2011-01-01")
EVAL_END = pd.Timestamp("2024-12-01")
CAL_END = pd.Timestamp("2010-12-31")
SEED = 7525
BOOTSTRAPS = 500


@dataclass(frozen=True)
class Variant:
    name: str
    met_column: str
    hyd_column: str
    onset_run: int = 3
    onset_threshold: float = -1.0
    termination_threshold: float = -0.5
    termination_run: int = 2
    window: int = 6


def lmom(values: np.ndarray) -> tuple[float, float, float]:
    values = np.sort(values)
    n = len(values)
    if n < 3:
        return np.nan, np.nan, np.nan
    index = np.arange(n)
    b0 = values.mean()
    b1 = np.sum(values * index) / (n * (n - 1))
    b2 = np.sum(values * index * (index - 1)) / (n * (n - 1) * (n - 2))
    return b0, 2 * b1 - b0, 6 * b2 - 6 * b1 + b0


def gamma_fit(values: np.ndarray) -> tuple[float, float] | None:
    values = values[values > 0]
    if len(values) < 5:
        return None
    l1, l2, _ = lmom(values)
    if l1 <= 0 or l2 <= 0:
        return None
    ratio = l2 / l1
    if ratio < 0.5:
        z = np.pi * ratio * ratio
        alpha = (1 - 0.3080 * z) / (z - 0.05812 * z * z + 0.01765 * z**3)
    else:
        z = 1 - ratio
        alpha = (0.7213 * z - 0.5947 * z * z) / (
            1 - 2.1817 * z + 1.2113 * z * z
        )
    return float(alpha), float(l1 / alpha)


def fixed_spi(accumulation: np.ndarray, months: np.ndarray, calibration: np.ndarray) -> np.ndarray:
    output = np.full(len(accumulation), np.nan)
    for month in range(1, 13):
        selected = months == month
        fitted = selected & calibration & np.isfinite(accumulation)
        values = accumulation[fitted]
        if len(values) < 10:
            continue
        zero_probability = float((values <= 0).mean())
        parameters = gamma_fit(values)
        if parameters is None:
            continue
        alpha, beta = parameters
        target = accumulation[selected]
        probability = np.where(
            target > 0,
            zero_probability
            + (1 - zero_probability)
            * special.gammainc(alpha, np.clip(target, 0, None) / beta),
            zero_probability / 2,
        )
        probability = np.clip(probability, 1e-6, 1 - 1e-6)
        probability = np.where(np.isfinite(target), probability, np.nan)
        output[selected] = stats.norm.ppf(probability)
    return np.clip(output, -4, 4)


def add_fixed_spi(series: pd.DataFrame) -> pd.DataFrame:
    """Calculate fixed-1980--2010 SPI without changing indices_v2.csv."""
    cache = OUT / "fixed_spi_1980_2010.parquet"
    if cache.exists():
        fixed = pd.read_parquet(cache)
        fixed["GAGE_ID"] = fixed.GAGE_ID.astype(str).str.zfill(8)
        fixed["date"] = pd.to_datetime(fixed.date)
        return series.merge(fixed, on=["GAGE_ID", "date"], how="left", validate="one_to_one")

    monthly = pd.read_csv(
        RES / "monthly_inputs.csv",
        usecols=["GAGE_ID", "date", "P_mm", "n_days_P_mm"],
        dtype={"GAGE_ID": str},
        parse_dates=["date"],
    )
    monthly["GAGE_ID"] = monthly.GAGE_ID.str.zfill(8)
    rows: list[pd.DataFrame] = []
    for count, (gage, group) in enumerate(monthly.groupby("GAGE_ID", sort=False), start=1):
        group = group.sort_values("date").copy()
        precipitation = group.P_mm.where(group.n_days_P_mm >= 25)
        months = group.date.dt.month.to_numpy()
        calibration = (group.date <= CAL_END).to_numpy()
        result = group[["GAGE_ID", "date"]].copy()
        for scale in (3, 6):
            accumulation = precipitation.rolling(scale, min_periods=scale).sum().to_numpy()
            result[f"SPI_{scale}_fixed"] = fixed_spi(accumulation, months, calibration)
        rows.append(result)
        if count % 600 == 0:
            print(f"  fixed SPI: {count} basins", flush=True)
    fixed = pd.concat(rows, ignore_index=True)
    fixed.to_parquet(cache, index=False, compression="zstd")
    return series.merge(fixed, on=["GAGE_ID", "date"], how="left", validate="one_to_one")


def detect_events(
    values: np.ndarray,
    dates: np.ndarray,
    onset_run: int,
    onset_threshold: float,
    termination_threshold: float,
    termination_run: int,
) -> list[dict[str, object]]:
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
            termination = cursor - 1 if right_censored else cursor - 1
            event_values = values[onset : termination + 1]
            finite = event_values[np.isfinite(event_values)]
            events.append(
                {
                    "onset_idx": onset,
                    "term_idx": termination,
                    "onset": dates[onset],
                    "termination": dates[termination],
                    "right_censored": right_censored,
                    "censored_by_missing": censored_by_missing,
                    "duration_months": termination - onset + 1,
                    "severity": float(finite.sum()) if len(finite) else np.nan,
                }
            )
            index = termination + 1
            continue
        index += 1
    return events


def containing_event(events: list[dict[str, object]], month_index: int) -> int | None:
    """Return the event containing an index, or None."""
    if not events:
        return None
    onsets = np.fromiter((int(event["onset_idx"]) for event in events), dtype=int)
    position = int(np.searchsorted(onsets, month_index, side="right") - 1)
    if position >= 0 and month_index <= int(events[position]["term_idx"]):
        return position
    return None


def eligibility_reason(
    event: dict[str, object],
    met_values: np.ndarray,
    hyd_values: np.ndarray,
    last_valid_hyd: int,
    window: int,
    onset_run: int,
) -> str:
    onset_date = pd.Timestamp(event["onset"])
    if not (EVAL_START <= onset_date <= EVAL_END):
        return "outside_evaluation_period"
    if bool(event["right_censored"]):
        return "unobserved_meteorological_termination"
    onset = int(event["onset_idx"])
    termination = int(event["term_idx"])
    # A candidate onset in the last association month still needs the full
    # onset run to be observable.  This prevents unequal right-edge opportunity.
    followup_end = termination + window + onset_run - 1
    if followup_end >= len(hyd_values) or followup_end > last_valid_hyd:
        return "insufficient_post_termination_followup"
    if not np.isfinite(met_values[onset : termination + 1]).all():
        return "missing_meteorological_index_inside_event"
    if not np.isfinite(hyd_values[onset : followup_end + 1]).all():
        return "missing_streamflow_index_inside_followup"
    return "eligible"


def admissible_edges(
    met_events: list[dict[str, object]],
    hyd_events: list[dict[str, object]],
    at_risk_indices: list[int],
    window: int,
) -> list[dict[str, int]]:
    edges: list[dict[str, int]] = []
    for met_index in at_risk_indices:
        meteorological = met_events[met_index]
        met_onset = int(meteorological["onset_idx"])
        met_termination = int(meteorological["term_idx"])
        for hyd_index, hydrologic in enumerate(hyd_events):
            hyd_onset = int(hydrologic["onset_idx"])
            lag = hyd_onset - met_onset
            if lag < 0 or hyd_onset > met_termination + window:
                continue
            overlap_start = max(met_onset, hyd_onset)
            overlap_end = min(met_termination, int(hydrologic["term_idx"]))
            overlap = max(0, overlap_end - overlap_start + 1)
            edges.append(
                {
                    "met_index": met_index,
                    "hyd_index": hyd_index,
                    "lag": lag,
                    "overlap": overlap,
                }
            )
    return edges


def maximum_one_to_one(
    met_indices: list[int], hyd_count: int, edges: list[dict[str, int]]
) -> list[dict[str, int]]:
    if not met_indices or not hyd_count or not edges:
        return []
    row_lookup = {met_index: row for row, met_index in enumerate(met_indices)}
    score = np.zeros((len(met_indices), hyd_count), dtype=float)
    lookup: dict[tuple[int, int], dict[str, int]] = {}
    for edge in edges:
        row = row_lookup[edge["met_index"]]
        column = edge["hyd_index"]
        score[row, column] = (
            1_000_000_000 + 1_000 * edge["overlap"] - edge["lag"]
        )
        lookup[(row, column)] = edge
    rows, columns = linear_sum_assignment(score, maximize=True)
    return [
        lookup[(row, column)]
        for row, column in zip(rows, columns)
        if (row, column) in lookup
    ]


def build_basin_catalogue(
    gage: str,
    timescale: int,
    group: pd.DataFrame,
    variant: Variant,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    dates = group.date.to_numpy()
    met_values = group[variant.met_column].to_numpy(dtype=float)
    hyd_values = group[variant.hyd_column].to_numpy(dtype=float)
    met_events = detect_events(
        met_values,
        dates,
        variant.onset_run,
        variant.onset_threshold,
        variant.termination_threshold,
        variant.termination_run,
    )
    hyd_events = detect_events(
        hyd_values,
        dates,
        variant.onset_run,
        variant.onset_threshold,
        variant.termination_threshold,
        variant.termination_run,
    )
    valid_hyd = np.flatnonzero(np.isfinite(hyd_values))
    last_valid_hyd = int(valid_hyd[-1]) if len(valid_hyd) else -1

    audit_rows: list[dict[str, object]] = []
    eligible_indices: list[int] = []
    preexisting: dict[int, int | None] = {}
    for met_index, event in enumerate(met_events):
        reason = eligibility_reason(
            event, met_values, hyd_values, last_valid_hyd, variant.window, variant.onset_run
        )
        onset_date = pd.Timestamp(event["onset"])
        if EVAL_START <= onset_date <= EVAL_END:
            audit_rows.append(
                {
                    "GAGE_ID": gage,
                    "timescale": timescale,
                    "variant": variant.name,
                    "met_onset": onset_date,
                    "eligibility": reason,
                }
            )
        if reason != "eligible":
            continue
        eligible_indices.append(met_index)
        containing = containing_event(hyd_events, int(event["onset_idx"]))
        if containing is not None and int(hyd_events[containing]["onset_idx"]) < int(
            event["onset_idx"]
        ):
            preexisting[met_index] = containing
        else:
            preexisting[met_index] = None

    at_risk = [index for index in eligible_indices if preexisting[index] is None]
    edges = admissible_edges(met_events, hyd_events, at_risk, variant.window)
    pairs = maximum_one_to_one(at_risk, len(hyd_events), edges)
    assigned = {pair["met_index"]: pair for pair in pairs}
    met_edge_count: dict[int, int] = {}
    hyd_edge_count: dict[int, int] = {}
    for edge in edges:
        met_edge_count[edge["met_index"]] = met_edge_count.get(edge["met_index"], 0) + 1
        hyd_edge_count[edge["hyd_index"]] = hyd_edge_count.get(edge["hyd_index"], 0) + 1

    rows: list[dict[str, object]] = []
    for met_index in eligible_indices:
        meteorological = met_events[met_index]
        met_onset_index = int(meteorological["onset_idx"])
        met_termination_index = int(meteorological["term_idx"])
        prior_hyd_index = preexisting[met_index]
        pair = assigned.get(met_index)
        n_admissible_onsets = met_edge_count.get(met_index, 0)
        unassigned_with_admissible_onset = (
            prior_hyd_index is None and pair is None and n_admissible_onsets > 0
        )

        if prior_hyd_index is not None:
            onset_pathway = "river_already_in_drought"
            assigned_hyd_index = None
            lag = np.nan
        elif pair is None:
            onset_pathway = "no_new_river_onset_within_window"
            assigned_hyd_index = None
            lag = np.nan
        else:
            assigned_hyd_index = pair["hyd_index"]
            lag = int(pair["lag"])
            onset_pathway = (
                "new_onset_by_confirmation"
                if lag <= variant.onset_run - 1
                else "new_onset_after_confirmation"
            )

        actual_hyd_index = containing_event(hyd_events, met_termination_index)
        actual_active = actual_hyd_index is not None
        if actual_active:
            termination_state = "river_active_at_met_termination"
        elif prior_hyd_index is not None:
            termination_state = "river_recovered_by_met_termination"
        elif assigned_hyd_index is None:
            termination_state = "no_matched_river_drought"
        else:
            assigned_event = hyd_events[assigned_hyd_index]
            termination_state = (
                "river_onset_pending"
                if int(assigned_event["onset_idx"]) > met_termination_index
                else "river_recovered_by_met_termination"
            )

        assigned_event = (
            hyd_events[assigned_hyd_index] if assigned_hyd_index is not None else None
        )
        actual_event = hyd_events[actual_hyd_index] if actual_hyd_index is not None else None
        prior_event = hyd_events[prior_hyd_index] if prior_hyd_index is not None else None

        if assigned_event is None:
            assigned_termination_state = "no_assigned_river_drought"
            assigned_termination_offset = np.nan
        elif (
            not bool(assigned_event["right_censored"])
            and int(assigned_event["term_idx"]) < met_termination_index
        ):
            assigned_termination_state = "assigned_event_recovered"
            assigned_termination_offset = int(assigned_event["term_idx"]) - met_termination_index
        elif int(assigned_event["onset_idx"]) > met_termination_index:
            assigned_termination_state = "assigned_onset_pending"
            assigned_termination_offset = (
                np.nan
                if bool(assigned_event["right_censored"])
                else int(assigned_event["term_idx"]) - met_termination_index
            )
        else:
            assigned_termination_state = "assigned_event_active"
            assigned_termination_offset = (
                np.nan
                if bool(assigned_event["right_censored"])
                else int(assigned_event["term_idx"]) - met_termination_index
            )

        rows.append(
            {
                "GAGE_ID": gage,
                "timescale": timescale,
                "variant": variant.name,
                "met_onset": meteorological["onset"],
                "met_termination": meteorological["termination"],
                "met_onset_idx": met_onset_index,
                "met_termination_idx": met_termination_index,
                "met_duration_months": meteorological["duration_months"],
                "met_severity": meteorological["severity"],
                "onset_pathway": onset_pathway,
                "river_preexisting_at_met_onset": prior_hyd_index is not None,
                "preexisting_hyd_onset": None if prior_event is None else prior_event["onset"],
                "preexisting_hyd_termination": None if prior_event is None else prior_event["termination"],
                "assigned_hyd_onset": None if assigned_event is None else assigned_event["onset"],
                "assigned_hyd_termination": None if assigned_event is None else assigned_event["termination"],
                "assigned_hyd_right_censored": None
                if assigned_event is None
                else bool(assigned_event["right_censored"]),
                "n_admissible_hydrologic_onsets": n_admissible_onsets,
                "unassigned_despite_admissible_onset": unassigned_with_admissible_onset,
                "assigned_hyd_candidate_origin_count": np.nan
                if assigned_hyd_index is None
                else hyd_edge_count.get(assigned_hyd_index, 0),
                "assigned_onset_lag_months": lag,
                "assigned_termination_offset_months": assigned_termination_offset,
                "assigned_termination_state": assigned_termination_state,
                "termination_state": termination_state,
                "river_active_at_met_termination": actual_active,
                "actual_hyd_onset": None if actual_event is None else actual_event["onset"],
                "actual_hyd_termination": None if actual_event is None else actual_event["termination"],
                "actual_hyd_right_censored": None
                if actual_event is None
                else bool(actual_event["right_censored"]),
                "actual_hyd_age_at_met_termination_months": np.nan
                if actual_event is None
                else met_termination_index - int(actual_event["onset_idx"]) + 1,
                "actual_remaining_duration_months": np.nan
                if actual_event is None or bool(actual_event["right_censored"])
                else int(actual_event["term_idx"]) - met_termination_index,
                "actual_recovery_event_time_months": np.nan
                if actual_event is None or bool(actual_event["right_censored"])
                else int(actual_event["term_idx"]) - met_termination_index + 1,
                "actual_censor_time_months": np.nan
                if actual_event is None or not bool(actual_event["right_censored"])
                else int(actual_event["term_idx"]) - met_termination_index,
            }
        )
    return rows, audit_rows


def load_attributes() -> pd.DataFrame:
    if BASIN_ATTRIBUTES.exists():
        merged = pd.read_csv(BASIN_ATTRIBUTES, dtype={"GAGE_ID": str, "HUC02": str})
        merged["GAGE_ID"] = merged.GAGE_ID.str.zfill(8)
        merged["HUC02"] = merged.HUC02.str.zfill(2)
        return merged[[
            "GAGE_ID", "HUC02", "AGGECOREGION", "CLASS", "aridity", "WTDEPAVE",
            "SNOW_PCT_PRECIP", "DRAIN_SQKM",
        ]]
    requested = {
        "conterm_basinid.txt": ["DRAIN_SQKM", "HUC02"],
        "conterm_climate.txt": ["PPTAVG_BASIN", "PET", "SNOW_PCT_PRECIP"],
        "conterm_soils.txt": ["WTDEPAVE"],
        "conterm_bas_classif.txt": ["AGGECOREGION", "CLASS"],
    }
    merged: pd.DataFrame | None = None
    for filename, columns in requested.items():
        data = pd.read_csv(
            GAGES / filename,
            dtype={"STAID": str, "HUC02": str},
            encoding="latin-1",
        )
        data["STAID"] = data.STAID.str.strip().str.zfill(8)
        available = [column for column in columns if column in data.columns]
        subset = data[["STAID", *available]].copy()
        merged = subset if merged is None else merged.merge(subset, on="STAID", how="outer")
    if merged is None:
        raise RuntimeError("No basin attributes loaded")
    merged.rename(columns={"STAID": "GAGE_ID"}, inplace=True)
    merged["HUC02"] = (
        merged.HUC02.astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(2)
    )
    precipitation_mm = pd.to_numeric(merged.PPTAVG_BASIN, errors="coerce") * 10.0
    merged["aridity"] = pd.to_numeric(merged.PET, errors="coerce") / precipitation_mm
    for column in ("WTDEPAVE", "SNOW_PCT_PRECIP", "aridity"):
        merged[column] = pd.to_numeric(merged[column], errors="coerce")
    return merged[
        [
            "GAGE_ID",
            "HUC02",
            "AGGECOREGION",
            "CLASS",
            "aridity",
            "WTDEPAVE",
            "SNOW_PCT_PRECIP",
        ]
    ]


def summarize_pathways(catalogue: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (variant, timescale), group in catalogue.groupby(["variant", "timescale"]):
        for stage, column in (
            ("onset", "onset_pathway"),
            ("termination", "termination_state"),
        ):
            counts = group[column].value_counts()
            for pathway, count in counts.items():
                rows.append(
                    {
                        "variant": variant,
                        "timescale": timescale,
                        "stage": stage,
                        "pathway": pathway,
                        "n_denominator": len(group),
                        "n_pathway": int(count),
                        "percent": 100 * count / len(group),
                    }
                )
    return pd.DataFrame(rows)


def first_run_start(
    values: np.ndarray, condition, run: int, max_start: int
) -> int | None:
    flags = np.asarray(condition(values), dtype=bool)
    for start in range(0, min(max_start, len(flags) - run + 1)):
        if flags[start : start + run].all():
            return start + 1
    return None


def classify_recovery(values: np.ndarray, horizon: int) -> str:
    recovery_start = first_run_start(
        values[:horizon], lambda x: x > -0.5, run=2, max_start=max(0, horizon - 1)
    )
    if recovery_start is None:
        return "no_confirmed_recovery"
    confirmation = recovery_start + 1
    recurrence = first_run_start(
        values[confirmation : horizon + 2],
        lambda x: x <= -1.0,
        run=3,
        max_start=max(0, horizon - confirmation),
    )
    return "temporary_recovery" if recurrence is not None else "sustained_recovery"


def recovery_confirmation_month(values: np.ndarray, horizon: int) -> int | None:
    """Return the second month of the first two-month recovery run, one-indexed."""
    recovery_start = first_run_start(
        values[:horizon], lambda x: x > -0.5, run=2, max_start=max(0, horizon - 1)
    )
    return None if recovery_start is None else recovery_start + 1


def recovery_catalogue(
    primary: pd.DataFrame, series: pd.DataFrame
) -> pd.DataFrame:
    active = primary[primary.river_active_at_met_termination].copy().reset_index(drop=True)
    active["event_id"] = np.arange(len(active))
    lookup = {
        (gage, int(scale)): indices.to_numpy()
        for (gage, scale), indices in active.groupby(["GAGE_ID", "timescale"]).groups.items()
    }
    rows: list[dict[str, object]] = []
    for gage, group in series.groupby("GAGE_ID", sort=False):
        group = group.sort_values("date").reset_index(drop=True)
        month_number = (group.date.dt.year * 12 + group.date.dt.month).to_numpy()
        positions = {int(value): index for index, value in enumerate(month_number)}
        for scale in (3, 6):
            event_indices = lookup.get((gage, scale))
            if event_indices is None:
                continue
            spei = group[f"SPEI_{scale}_fixed"].to_numpy(dtype=float)
            ssi = group[f"SSI_{scale}_fixed"].to_numpy(dtype=float)
            for event_index in event_indices:
                event = active.loc[event_index]
                term = pd.Timestamp(event.met_termination)
                term_number = term.year * 12 + term.month
                start = positions.get(term_number + 1)
                base = {
                    "event_id": int(event.event_id),
                    "GAGE_ID": gage,
                    "timescale": scale,
                    "met_onset": event.met_onset,
                    "met_termination": event.met_termination,
                    "onset_pathway": event.onset_pathway,
                    "actual_hyd_onset": event.actual_hyd_onset,
                    "actual_hyd_termination": event.actual_hyd_termination,
                }
                for horizon in (3, 6, 9, 12):
                    if start is None:
                        rows.append(
                            {**base, "horizon_months": horizon, "status": "incomplete_record"}
                        )
                        continue
                    needed = horizon + 2
                    spei_path = spei[start : start + needed]
                    ssi_path = ssi[start : start + needed]
                    complete = (
                        len(spei_path) == needed
                        and len(ssi_path) == needed
                        and np.isfinite(spei_path).all()
                        and np.isfinite(ssi_path).all()
                    )
                    if not complete:
                        rows.append(
                            {**base, "horizon_months": horizon, "status": "incomplete_record"}
                        )
                        continue
                    new_met = first_run_start(
                        spei_path,
                        lambda x: x <= -1.0,
                        run=3,
                        max_start=horizon,
                    )
                    trajectory = classify_recovery(ssi_path, horizon)
                    recovery_confirmation = recovery_confirmation_month(ssi_path, horizon)
                    if new_met is None:
                        reforcing_timing = None
                    elif (
                        recovery_confirmation is None
                        or new_met <= recovery_confirmation
                    ):
                        reforcing_timing = "new_forcing_before_confirmed_river_recovery"
                    else:
                        reforcing_timing = "new_forcing_after_confirmed_river_recovery"
                    status = (
                        "renewed_forcing_before_recovery"
                        if reforcing_timing == "new_forcing_before_confirmed_river_recovery"
                        else trajectory
                    )
                    rows.append(
                        {
                            **base,
                            "horizon_months": horizon,
                            "status": status,
                            "trajectory_ignoring_reforcing": trajectory,
                            "new_met_onset_month": new_met,
                            "river_recovery_confirmation_month": recovery_confirmation,
                            "reforcing_timing": reforcing_timing,
                        }
                    )
    return pd.DataFrame(rows)


def summarize_recovery(recovery: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    full_rows: list[dict[str, object]] = []
    conditional_rows: list[dict[str, object]] = []
    for (timescale, horizon), group in recovery.groupby(["timescale", "horizon_months"]):
        complete = group[group.status.ne("incomplete_record")]
        counts = complete.status.value_counts()
        for outcome in (
            "sustained_recovery",
            "temporary_recovery",
            "no_confirmed_recovery",
            "renewed_forcing_before_recovery",
        ):
            count = int(counts.get(outcome, 0))
            full_rows.append(
                {
                    "timescale": timescale,
                    "horizon_months": horizon,
                    "active_events": len(group),
                    "complete_events": len(complete),
                    "incomplete_events": int((group.status == "incomplete_record").sum()),
                    "outcome": outcome,
                    "n_outcome": count,
                    "percent_of_complete": 100 * count / len(complete) if len(complete) else np.nan,
                    "percent_of_all_active": 100 * count / len(group) if len(group) else np.nan,
                }
            )
        uninterrupted = complete[
            complete.status.ne("renewed_forcing_before_recovery")
        ]
        conditional_counts = uninterrupted.status.value_counts()
        for outcome in (
            "sustained_recovery",
            "temporary_recovery",
            "no_confirmed_recovery",
        ):
            count = int(conditional_counts.get(outcome, 0))
            conditional_rows.append(
                {
                    "timescale": timescale,
                    "horizon_months": horizon,
                    "eligible_uninterrupted_events": len(uninterrupted),
                    "outcome": outcome,
                    "n_outcome": count,
                    "percent": 100 * count / len(uninterrupted)
                    if len(uninterrupted)
                    else np.nan,
                }
            )
    return pd.DataFrame(full_rows), pd.DataFrame(conditional_rows)


def weighted_km(
    time: np.ndarray, observed: np.ndarray, weights: np.ndarray, grid: np.ndarray
) -> np.ndarray:
    survival = 1.0
    output = np.ones(len(grid), dtype=float)
    cursor = 0
    event_times = np.unique(time[observed & (weights > 0)])
    for event_time in event_times:
        while cursor < len(grid) and grid[cursor] < event_time:
            output[cursor] = survival
            cursor += 1
        at_risk = weights[time >= event_time].sum()
        if at_risk > 0:
            survival *= 1.0 - weights[(time == event_time) & observed].sum() / at_risk
        while cursor < len(grid) and grid[cursor] == event_time:
            output[cursor] = survival
            cursor += 1
    output[cursor:] = survival
    return output


def operational_persistence(primary: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    rng = np.random.default_rng(SEED)
    grid = np.arange(0, 25)
    for timescale in (3, 6):
        active = primary[
            primary.timescale.eq(timescale) & primary.river_active_at_met_termination
        ].copy()
        observed = active.actual_recovery_event_time_months.notna().to_numpy()
        time = np.where(
            observed,
            active.actual_recovery_event_time_months,
            active.actual_censor_time_months,
        ).astype(int)
        basins = np.sort(active.GAGE_ID.unique())
        basin_row = active.GAGE_ID.map({gage: index for index, gage in enumerate(basins)}).to_numpy()
        survival = weighted_km(time, observed, np.ones(len(active)), grid)
        bootstrap = np.empty((BOOTSTRAPS, len(grid)), dtype=float)
        for draw in range(BOOTSTRAPS):
            weights = np.bincount(
                rng.integers(0, len(basins), len(basins)), minlength=len(basins)
            )[basin_row]
            bootstrap[draw] = weighted_km(time, observed, weights, grid)
        for elapsed in range(0, 19):
            ratio = survival[elapsed + 6] / survival[elapsed]
            draws = bootstrap[:, elapsed + 6] / bootstrap[:, elapsed]
            rows.append(
                {
                    "timescale": timescale,
                    "months_already_active_after_met_termination": elapsed,
                    "additional_horizon_months": 6,
                    "probability_active_six_more_months": ratio,
                    "ci_low": float(np.nanquantile(draws, 0.025)),
                    "ci_high": float(np.nanquantile(draws, 0.975)),
                    "n_at_risk": int((time >= elapsed).sum()),
                }
            )
    return pd.DataFrame(rows)


def pair_only_bias(primary: pd.DataFrame, attributes: pd.DataFrame) -> pd.DataFrame:
    frame = primary.merge(attributes, on="GAGE_ID", how="left", validate="many_to_one")
    rows: list[dict[str, object]] = []
    regional_definitions = {
        "United States": pd.Series("United States", index=frame.index),
        "aggregate_ecoregion": frame.AGGECOREGION,
        "HUC02": frame.HUC02,
    }
    for region_type, labels in regional_definitions.items():
        work = frame.assign(region=labels)
        for (timescale, region), group in work.groupby(["timescale", "region"], dropna=False):
            at_risk = group[~group.river_preexisting_at_met_onset]
            paired = at_risk[at_risk.assigned_onset_lag_months.notna()]
            conflicted = at_risk[at_risk.unassigned_despite_admissible_onset]
            no_admissible = at_risk[
                at_risk.assigned_onset_lag_months.isna()
                & ~at_risk.unassigned_despite_admissible_onset
            ]
            observed_pair_termination = paired[
                paired.assigned_termination_offset_months.notna()
            ]
            actual_active = group[group.river_active_at_met_termination]
            actual_active_paired = paired[paired.river_active_at_met_termination]
            assigned_active_paired = paired[
                paired.assigned_termination_state.eq("assigned_event_active")
            ]
            full_propagation = len(paired) / len(at_risk) if len(at_risk) else np.nan
            actual_active_share_paired = (
                len(actual_active_paired) / len(paired) if len(paired) else np.nan
            )
            assigned_active_share_paired = (
                len(assigned_active_paired) / len(paired) if len(paired) else np.nan
            )
            pooled_offset = observed_pair_termination.assigned_termination_offset_months.median()
            active_remaining = actual_active.actual_remaining_duration_months.median()
            rows.append(
                {
                    "region_type": region_type,
                    "region": region,
                    "timescale": timescale,
                    "n_all_meteorological_events": len(group),
                    "n_river_already_in_drought": int(group.river_preexisting_at_met_onset.sum()),
                    "n_at_risk_for_new_onset": len(at_risk),
                    "n_pairs_retained_by_pair_only": len(paired),
                    "pair_only_conditioned_on_match_percent": 100.0 if len(paired) else np.nan,
                    "full_denominator_new_onset_percent": 100 * full_propagation,
                    "at_risk_events_excluded_by_pair_only_percent": 100 * (1 - full_propagation),
                    "n_unassigned_despite_admissible_onset": len(conflicted),
                    "unassigned_despite_admissible_onset_percent_of_at_risk": 100
                    * len(conflicted)
                    / len(at_risk)
                    if len(at_risk)
                    else np.nan,
                    "n_without_any_admissible_onset": len(no_admissible),
                    "without_any_admissible_onset_percent_of_at_risk": 100
                    * len(no_admissible)
                    / len(at_risk)
                    if len(at_risk)
                    else np.nan,
                    "pair_only_onset_lag_median_months": paired.assigned_onset_lag_months.median(),
                    "pair_only_pooled_termination_offset_median_months": pooled_offset,
                    "assigned_event_active_percent_among_pairs": 100 * assigned_active_share_paired,
                    "actual_river_active_percent_among_pairs": 100 * actual_active_share_paired,
                    "assigned_status_understatement_percentage_points": 100
                    * (actual_active_share_paired - assigned_active_share_paired),
                    "actual_river_active_percent_all_events": 100
                    * len(actual_active)
                    / len(group),
                    "actual_active_remaining_duration_median_months": active_remaining,
                    "active_minus_pooled_offset_month_contrast": active_remaining - pooled_offset,
                    "contrast_note": "The last contrast compares different estimands: remaining duration among rivers actually active versus the pooled signed offset among completed assigned pairs.",
                }
            )
    return pd.DataFrame(rows)


def matching_ambiguity_summary(primary: pd.DataFrame) -> pd.DataFrame:
    """Separate lack of an observed onset from one-to-one assignment conflict."""
    rows: list[dict[str, object]] = []
    for timescale, group in primary.groupby("timescale"):
        at_risk = group[~group.river_preexisting_at_met_onset]
        assigned = at_risk[at_risk.assigned_onset_lag_months.notna()]
        conflicted = at_risk[at_risk.unassigned_despite_admissible_onset]
        no_admissible = at_risk[
            at_risk.assigned_onset_lag_months.isna()
            & ~at_risk.unassigned_despite_admissible_onset
        ]
        assigned_competing = assigned[
            assigned.assigned_hyd_candidate_origin_count.gt(1)
        ]
        for status, subset in (
            ("assigned_new_river_onset", assigned),
            ("unassigned_due_to_one_to_one_competition", conflicted),
            ("no_admissible_new_river_onset_observed", no_admissible),
        ):
            rows.append(
                {
                    "timescale": timescale,
                    "status": status,
                    "n_at_risk": len(at_risk),
                    "n_events": len(subset),
                    "percent_of_at_risk": 100 * len(subset) / len(at_risk),
                }
            )
        rows.append(
            {
                "timescale": timescale,
                "status": "assigned_onset_also_candidate_for_multiple_origins",
                "n_at_risk": len(at_risk),
                "n_events": len(assigned_competing),
                "percent_of_at_risk": 100 * len(assigned_competing) / len(at_risk),
            }
        )
    return pd.DataFrame(rows)


def fit_beta_prior(y: pd.Series, n: pd.Series) -> tuple[float, float]:
    frame = pd.concat([y.rename("y"), n.rename("n")], axis=1).dropna()
    frame = frame[(frame.n > 0) & (frame.y >= 0) & (frame.y <= frame.n)]
    yy = frame.y.to_numpy(dtype=float)
    nn = frame.n.to_numpy(dtype=float)
    pooled = float(yy.sum() / nn.sum())

    def objective(log_parameters: np.ndarray) -> float:
        alpha, beta = np.exp(log_parameters)
        return float(-np.sum(betaln(yy + alpha, nn - yy + beta) - betaln(alpha, beta)))

    concentration = 8.0
    start = np.log(
        [max(pooled * concentration, 0.05), max((1 - pooled) * concentration, 0.05)]
    )
    result = minimize(objective, start, method="Nelder-Mead", options={"maxiter": 5000})
    if not result.success:
        raise RuntimeError(f"Beta-binomial fit failed: {result.message}")
    alpha, beta = np.exp(result.x)
    return float(alpha), float(beta)


def partially_pooled_groups(
    frame: pd.DataFrame,
    group_column: str,
    outcome_column: str,
    denominator_label: str,
    feature: str,
    timescale: int,
) -> list[dict[str, object]]:
    basin_counts = (
        frame.groupby("GAGE_ID")[outcome_column]
        .agg(["sum", "count"])
        .rename(columns={"sum": "y", "count": "n"})
    )
    alpha, beta = fit_beta_prior(basin_counts.y, basin_counts.n)
    grouped = frame.groupby(group_column)[outcome_column].agg(["sum", "count"])
    rows: list[dict[str, object]] = []
    for label, values in grouped.iterrows():
        y = int(values["sum"])
        n = int(values["count"])
        post_alpha = alpha + y
        post_beta = beta + n - y
        subset = frame[frame[group_column].eq(label)]
        names = subset.GAGE_ID.drop_duplicates().to_numpy()
        locations = {
            name: subset.index[subset.GAGE_ID.eq(name)].to_numpy() for name in names
        }
        rng = np.random.default_rng(
            SEED + timescale * 1009 + sum(ord(character) for character in f"{feature}:{outcome_column}:{label}")
        )
        draws = np.empty(BOOTSTRAPS, dtype=float)
        for draw in range(BOOTSTRAPS):
            sampled = rng.choice(names, size=len(names), replace=True)
            indices = np.concatenate([locations[name] for name in sampled])
            draws[draw] = frame.loc[indices, outcome_column].mean()
        rows.append(
            {
                "timescale": timescale,
                "feature": feature,
                "group": label,
                "outcome": outcome_column,
                "denominator": denominator_label,
                "n_events": n,
                "n_outcome": y,
                "raw_percent": 100 * y / n,
                "posterior_percent": 100 * post_alpha / (post_alpha + post_beta),
                "ci_low_percent": 100 * np.quantile(draws, 0.025),
                "ci_high_percent": 100 * np.quantile(draws, 0.975),
                "interval_method": "basin-cluster bootstrap",
                "prior_alpha": alpha,
                "prior_beta": beta,
            }
        )
    return rows


def stratification(
    primary: pd.DataFrame,
    recovery: pd.DataFrame,
    attributes: pd.DataFrame,
) -> pd.DataFrame:
    frame = primary.merge(attributes, on="GAGE_ID", how="left", validate="many_to_one")
    recovery12 = recovery[recovery.horizon_months.eq(12)].merge(
        attributes, on="GAGE_ID", how="left", validate="many_to_one"
    )
    rows: list[dict[str, object]] = []
    for feature in ("aridity", "WTDEPAVE", "SNOW_PCT_PRECIP"):
        basin_values = attributes[["GAGE_ID", feature]].dropna().drop_duplicates("GAGE_ID")
        basin_values["decile"] = (
            pd.qcut(basin_values[feature], 10, labels=False, duplicates="drop") + 1
        )
        decile = basin_values.set_index("GAGE_ID").decile
        for timescale in (3, 6):
            scale = frame[frame.timescale.eq(timescale)].copy()
            scale["decile"] = scale.GAGE_ID.map(decile)
            scale = scale.dropna(subset=["decile"])
            scale["decile"] = scale.decile.astype(int)
            scale["river_already_in_drought"] = scale.river_preexisting_at_met_onset.astype(int)
            rows.extend(
                partially_pooled_groups(
                    scale,
                    "decile",
                    "river_already_in_drought",
                    "all eligible meteorological droughts",
                    feature,
                    timescale,
                )
            )
            at_risk = scale[~scale.river_preexisting_at_met_onset].copy()
            at_risk["no_observed_new_river_onset"] = at_risk[
                "n_admissible_hydrologic_onsets"
            ].eq(0).astype(int)
            rows.extend(
                partially_pooled_groups(
                    at_risk,
                    "decile",
                    "no_observed_new_river_onset",
                    "meteorological droughts with river not already in drought",
                    feature,
                    timescale,
                )
            )
            scale["river_active_at_termination"] = scale.river_active_at_met_termination.astype(int)
            rows.extend(
                partially_pooled_groups(
                    scale,
                    "decile",
                    "river_active_at_termination",
                    "all eligible meteorological droughts",
                    feature,
                    timescale,
                )
            )

            recovered = recovery12[
                recovery12.timescale.eq(timescale)
                & recovery12.status.ne("incomplete_record")
            ].copy()
            recovered["decile"] = recovered.GAGE_ID.map(decile)
            recovered = recovered.dropna(subset=["decile"])
            recovered["decile"] = recovered.decile.astype(int)
            recovered["re_forced_within_12_months"] = recovered.reforcing_timing.notna().astype(int)
            rows.extend(
                partially_pooled_groups(
                    recovered,
                    "decile",
                    "re_forced_within_12_months",
                    "actual-active events with complete 12-month record",
                    feature,
                    timescale,
                )
            )
            uninterrupted = recovered[
                recovered.status.ne("renewed_forcing_before_recovery")
            ].copy()
            uninterrupted["no_confirmed_recovery_12_months"] = uninterrupted.status.eq(
                "no_confirmed_recovery"
            ).astype(int)
            rows.extend(
                partially_pooled_groups(
                    uninterrupted,
                    "decile",
                    "no_confirmed_recovery_12_months",
                    "actual-active events with uninterrupted meteorological recovery",
                    feature,
                    timescale,
                )
            )

    # Reference versus non-reference uses the same partial-pooling calculation.
    for timescale in (3, 6):
        scale = frame[frame.timescale.eq(timescale)].dropna(subset=["CLASS"]).copy()
        scale["river_already_in_drought"] = scale.river_preexisting_at_met_onset.astype(int)
        rows.extend(
            partially_pooled_groups(
                scale,
                "CLASS",
                "river_already_in_drought",
                "all eligible meteorological droughts",
                "reference_class",
                timescale,
            )
        )
        at_risk = scale[~scale.river_preexisting_at_met_onset].copy()
        at_risk["no_observed_new_river_onset"] = at_risk[
            "n_admissible_hydrologic_onsets"
        ].eq(0).astype(int)
        rows.extend(
            partially_pooled_groups(
                at_risk,
                "CLASS",
                "no_observed_new_river_onset",
                "meteorological droughts with river not already in drought",
                "reference_class",
                timescale,
            )
        )
        scale["river_active_at_termination"] = scale.river_active_at_met_termination.astype(int)
        rows.extend(
            partially_pooled_groups(
                scale,
                "CLASS",
                "river_active_at_termination",
                "all eligible meteorological droughts",
                "reference_class",
                timescale,
            )
        )
    return pd.DataFrame(rows)


def truncation_sensitivity(primary: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    frame = primary.copy()
    frame["met_onset"] = pd.to_datetime(frame.met_onset)
    for label, subset in (
        ("all_2011_2024", frame),
        ("onsets_through_2021", frame[frame.met_onset <= "2021-12-01"]),
    ):
        for timescale, group in subset.groupby("timescale"):
            for stage, column in (
                ("onset", "onset_pathway"),
                ("termination", "termination_state"),
            ):
                counts = group[column].value_counts()
                for pathway, count in counts.items():
                    rows.append(
                        {
                            "cohort": label,
                            "timescale": timescale,
                            "stage": stage,
                            "pathway": pathway,
                            "n_events": len(group),
                            "n_pathway": int(count),
                            "percent": 100 * count / len(group),
                        }
                    )
    return pd.DataFrame(rows)


def primary_audit(primary: pd.DataFrame) -> dict[str, object]:
    duplicate_met = int(primary.duplicated(["GAGE_ID", "timescale", "met_onset"]).sum())
    paired = primary[primary.assigned_hyd_onset.notna()]
    duplicate_hyd = int(paired.duplicated(["GAGE_ID", "timescale", "assigned_hyd_onset"]).sum())
    preexisting_with_pair = int(
        (primary.river_preexisting_at_met_onset & primary.assigned_hyd_onset.notna()).sum()
    )
    active_state_failures = int(
        (
            primary.termination_state.eq("river_active_at_met_termination")
            != primary.river_active_at_met_termination
        ).sum()
    )
    return {
        "n_events": len(primary),
        "duplicate_meteorological_origins": duplicate_met,
        "duplicate_assigned_hydrologic_events": duplicate_hyd,
        "preexisting_origins_with_new_onset_assignment": preexisting_with_pair,
        "actual_active_state_logic_failures": active_state_failures,
    }


def main() -> None:
    print("Loading fixed-calibration drought indices...", flush=True)
    series = pd.read_parquet(RES / "audited_indices.parquet")
    series = series[series.primary_population].copy()
    series["GAGE_ID"] = series.GAGE_ID.astype(str).str.zfill(8)
    series["date"] = pd.to_datetime(series.date)

    variants = [
        Variant("primary", "SPEI_{scale}_fixed", "SSI_{scale}_fixed"),
        Variant("full_period_calibration", "SPEI_{scale}", "SSI_{scale}"),
        Variant("spi_fixed_calibration", "SPI_{scale}_fixed", "SSI_{scale}_fixed"),
        Variant("spi_full_period", "SPI_{scale}", "SSI_{scale}"),
        Variant("ssi_1_month", "SPEI_{scale}_fixed", "SSI_1_fixed"),
        Variant("ssi_empirical", "SPEI_{scale}_fixed", "SSI_{scale}_empirical_fixed"),
        Variant("onset_threshold_-0.8", "SPEI_{scale}_fixed", "SSI_{scale}_fixed", onset_threshold=-0.8),
        Variant("onset_threshold_-1.2", "SPEI_{scale}_fixed", "SSI_{scale}_fixed", onset_threshold=-1.2),
        Variant("termination_threshold_0.0", "SPEI_{scale}_fixed", "SSI_{scale}_fixed", termination_threshold=0.0),
        Variant("termination_threshold_-0.75", "SPEI_{scale}_fixed", "SSI_{scale}_fixed", termination_threshold=-0.75),
        Variant("matching_window_3", "SPEI_{scale}_fixed", "SSI_{scale}_fixed", window=3),
        Variant("matching_window_9", "SPEI_{scale}_fixed", "SSI_{scale}_fixed", window=9),
        Variant("matching_window_12", "SPEI_{scale}_fixed", "SSI_{scale}_fixed", window=12),
        Variant("onset_persistence_1", "SPEI_{scale}_fixed", "SSI_{scale}_fixed", onset_run=1),
        Variant("onset_persistence_2", "SPEI_{scale}_fixed", "SSI_{scale}_fixed", onset_run=2),
        Variant("recovery_persistence_1", "SPEI_{scale}_fixed", "SSI_{scale}_fixed", termination_run=1),
        Variant("recovery_persistence_3", "SPEI_{scale}_fixed", "SSI_{scale}_fixed", termination_run=3),
    ]

    all_rows: list[dict[str, object]] = []
    audit_rows: list[dict[str, object]] = []
    grouped = list(series.groupby("GAGE_ID", sort=False))
    for variant in variants:
        print(f"Variant: {variant.name}", flush=True)
        for count, (gage, group) in enumerate(grouped, start=1):
            ordered = group.sort_values("date").reset_index(drop=True)
            for scale in (3, 6):
                resolved = Variant(
                    variant.name,
                    variant.met_column.format(scale=scale),
                    variant.hyd_column.format(scale=scale),
                    variant.onset_run,
                    variant.onset_threshold,
                    variant.termination_threshold,
                    variant.termination_run,
                    variant.window,
                )
                rows, audit = build_basin_catalogue(gage, scale, ordered, resolved)
                all_rows.extend(rows)
                if variant.name == "primary":
                    audit_rows.extend(audit)
            if count % 900 == 0:
                print(f"  {count} basins", flush=True)

    catalogue = pd.DataFrame(all_rows)
    for column in (
        "met_onset",
        "met_termination",
        "preexisting_hyd_onset",
        "preexisting_hyd_termination",
        "assigned_hyd_onset",
        "assigned_hyd_termination",
        "actual_hyd_onset",
        "actual_hyd_termination",
    ):
        catalogue[column] = pd.to_datetime(catalogue[column])
    catalogue.to_parquet(OUT / "revised_state_catalogue_all_variants.parquet", index=False, compression="zstd")
    summary = summarize_pathways(catalogue)
    summary.to_csv(OUT / "revised_pathway_robustness_summary.csv", index=False)

    primary = catalogue[catalogue.variant.eq("primary")].copy().reset_index(drop=True)
    primary.to_parquet(OUT / "revised_state_catalogue_primary.parquet", index=False, compression="zstd")
    audit = pd.DataFrame(audit_rows)
    audit.to_csv(OUT / "meteorological_event_eligibility_flow.csv", index=False)
    eligibility_summary = (
        audit.groupby(["timescale", "eligibility"]).size().rename("n_events").reset_index()
    )
    eligibility_summary.to_csv(OUT / "meteorological_event_eligibility_summary.csv", index=False)
    qa = primary_audit(primary)
    (OUT / "revised_catalogue_audit.json").write_text(json.dumps(qa, indent=2) + "\n")
    if any(qa[key] for key in qa if key != "n_events"):
        raise RuntimeError(f"Revised catalogue audit failed: {qa}")

    print("Classifying full-denominator recovery outcomes...", flush=True)
    recovery = recovery_catalogue(primary, series)
    recovery.to_parquet(OUT / "revised_recovery_event_catalogue.parquet", index=False, compression="zstd")
    recovery_full, recovery_conditional = summarize_recovery(recovery)
    recovery_full.to_csv(OUT / "recovery_full_denominator_summary.csv", index=False)
    recovery_conditional.to_csv(OUT / "recovery_uninterrupted_conditional_summary.csv", index=False)
    reforced = recovery[recovery.reforcing_timing.notna()].copy()
    reforcing_summary = (
        reforced.groupby(["timescale", "horizon_months", "reforcing_timing"])
        .size()
        .rename("n_events")
        .reset_index()
    )
    reforcing_totals = reforced.groupby(["timescale", "horizon_months"]).size()
    reforcing_summary["percent_of_reforced_events"] = [
        100 * row.n_events / reforcing_totals.loc[(row.timescale, row.horizon_months)]
        for row in reforcing_summary.itertuples()
    ]
    reforcing_summary.to_csv(OUT / "reforcing_timing_summary.csv", index=False)

    print("Calculating pair-only contrasts and hydrological stratification...", flush=True)
    attributes = load_attributes()
    pair_bias = pair_only_bias(primary, attributes)
    pair_bias.to_csv(OUT / "pair_only_vs_full_denominator_by_region.csv", index=False)
    matching_ambiguity_summary(primary).to_csv(
        OUT / "matching_ambiguity_summary.csv", index=False
    )
    stratified = stratification(primary, recovery, attributes)
    stratified.to_csv(OUT / "partially_pooled_hydrological_stratification.csv", index=False)
    truncation_sensitivity(primary).to_csv(OUT / "truncation_sensitivity.csv", index=False)
    persistence = operational_persistence(primary)
    persistence.to_csv(OUT / "operational_six_month_persistence.csv", index=False)

    provenance = {
        "primary_meteorological_index": "SPEI using basin-mean gridMET reference evapotranspiration extracted from the gridMET pet product",
        "gridmet_pet_method": "gridMET documentation identifies reference evapotranspiration as ASCE Penman-Monteith; the local downloader and extractor used gridMET pet files, not a local Thornthwaite calculation",
        "fixed_spi_added": True,
        "output_directory": str(OUT),
    }
    (OUT / "analysis_provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(f"Analysis outputs written to {OUT}", flush=True)


if __name__ == "__main__":
    main()
