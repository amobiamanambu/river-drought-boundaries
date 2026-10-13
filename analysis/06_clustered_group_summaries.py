#!/usr/bin/env python3
"""Basin-cluster uncertainty for regional and reference-class contrasts."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from _paths import BASIN_ATTRIBUTES, OUTPUTS


OUT = OUTPUTS
SEED = 7525
BOOTSTRAPS = 500


def load_framework():
    path = Path(__file__).with_name("02_run_audited_state_framework.py")
    spec = importlib.util.spec_from_file_location("audited_framework_groups", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def clustered_share(frame: pd.DataFrame, outcome: str, seed: int) -> tuple[float, float, float, int]:
    grouped = frame.groupby("GAGE_ID")[outcome].agg(["sum", "count"])
    numerator = grouped["sum"].to_numpy(float)
    denominator = grouped["count"].to_numpy(float)
    point = numerator.sum() / denominator.sum()
    rng = np.random.default_rng(seed)
    draws = np.empty(BOOTSTRAPS)
    for draw in range(BOOTSTRAPS):
        selected = rng.integers(0, len(grouped), len(grouped))
        draws[draw] = numerator[selected].sum() / denominator[selected].sum()
    low, high = np.quantile(draws, [0.025, 0.975])
    return 100 * point, 100 * low, 100 * high, len(grouped)


def grouped_rows(
    frame: pd.DataFrame,
    group_column: str,
    outcome: str,
    denominator: str,
) -> list[dict[str, object]]:
    rows = []
    for (timescale, group), subset in frame.groupby(["timescale", group_column], dropna=False):
        if not len(subset):
            continue
        point, low, high, basins = clustered_share(
            subset, outcome, SEED + int(timescale) * 100 + sum(ord(x) for x in str(group))
        )
        rows.append({
            "grouping": group_column,
            "group": group,
            "timescale": int(timescale),
            "outcome": outcome,
            "denominator": denominator,
            "n_events": len(subset),
            "n_basins": basins,
            "percent": point,
            "cluster_ci_low_percent": low,
            "cluster_ci_high_percent": high,
            "n_bootstrap": BOOTSTRAPS,
        })
    return rows


def main() -> None:
    framework = load_framework()
    primary = pd.read_parquet(OUT / "revised_state_catalogue_primary.parquet")
    recovery = pd.read_parquet(OUT / "revised_recovery_event_catalogue.parquet")
    attributes = framework.load_attributes()
    attributes["reference_class"] = np.where(attributes.CLASS.eq("Ref"), "Reference", "Non-reference")

    events = primary.merge(
        attributes[["GAGE_ID", "HUC02", "AGGECOREGION", "reference_class"]],
        on="GAGE_ID", how="left", validate="many_to_one",
    )
    events["United States"] = "United States"
    events["river_already_in_drought"] = events.river_preexisting_at_met_onset.astype(int)
    events["river_active_at_termination"] = events.river_active_at_met_termination.astype(int)
    at_risk = events[~events.river_preexisting_at_met_onset].copy()
    at_risk["no_admissible_new_onset"] = at_risk.n_admissible_hydrologic_onsets.eq(0).astype(int)

    recovery12 = recovery[
        recovery.horizon_months.eq(12) & recovery.status.ne("incomplete_record")
    ].merge(
        attributes[["GAGE_ID", "HUC02", "AGGECOREGION", "reference_class"]],
        on="GAGE_ID", how="left", validate="many_to_one",
    )
    recovery12["United States"] = "United States"
    recovery12["neither_recovery_nor_renewed_forcing"] = recovery12.status.eq(
        "no_confirmed_recovery"
    ).astype(int)
    recovery12["renewed_forcing_before_recovery"] = recovery12.status.eq(
        "renewed_forcing_before_recovery"
    ).astype(int)

    rows: list[dict[str, object]] = []
    for grouping in ("United States", "HUC02", "AGGECOREGION", "reference_class"):
        rows.extend(grouped_rows(events, grouping, "river_already_in_drought", "all eligible meteorological droughts"))
        rows.extend(grouped_rows(events, grouping, "river_active_at_termination", "all eligible meteorological droughts"))
        rows.extend(grouped_rows(at_risk, grouping, "no_admissible_new_onset", "initially at-risk meteorological droughts"))
        rows.extend(grouped_rows(recovery12, grouping, "neither_recovery_nor_renewed_forcing", "active boundary encounters with complete 12-month follow-up"))
        rows.extend(grouped_rows(recovery12, grouping, "renewed_forcing_before_recovery", "active boundary encounters with complete 12-month follow-up"))
    pd.DataFrame(rows).to_csv(OUT / "clustered_regional_reference_summary.csv", index=False)

    basin_id = pd.read_csv(
        BASIN_ATTRIBUTES,
        usecols=["GAGE_ID", "DRAIN_SQKM"], dtype={"GAGE_ID": str},
    )
    basin_id["GAGE_ID"] = basin_id.GAGE_ID.str.zfill(8)
    area_events = events.merge(basin_id, on="GAGE_ID", how="left", validate="many_to_one")
    area_rows = []
    for label, minimum in (("all_primary_basins", 0), ("area_at_least_16_km2", 16), ("area_at_least_100_km2", 100)):
        subset = area_events[pd.to_numeric(area_events.DRAIN_SQKM, errors="coerce").ge(minimum)].copy()
        for timescale, scale in subset.groupby("timescale"):
            at_risk_scale = scale[~scale.river_preexisting_at_met_onset]
            area_rows.append({
                "cohort": label,
                "minimum_drainage_area_km2": minimum,
                "timescale": int(timescale),
                "n_basins": int(scale.GAGE_ID.nunique()),
                "n_events": len(scale),
                "preexisting_percent": 100 * scale.river_preexisting_at_met_onset.mean(),
                "active_at_termination_percent": 100 * scale.river_active_at_met_termination.mean(),
                "no_admissible_onset_percent_of_at_risk": 100 * at_risk_scale.n_admissible_hydrologic_onsets.eq(0).mean(),
            })
    pd.DataFrame(area_rows).to_csv(OUT / "basin_area_sensitivity.csv", index=False)
    print("Saved clustered regional and reference-class summaries.")


if __name__ == "__main__":
    main()
