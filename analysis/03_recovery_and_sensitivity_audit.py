#!/usr/bin/env python3
"""Finish the audited rebuild with cohort and unique-event sensitivities."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from _paths import DATA_ROOT, OUTPUTS


OUT = OUTPUTS


def load_framework():
    path = Path(__file__).with_name("02_run_audited_state_framework.py")
    spec = importlib.util.spec_from_file_location("audited_framework", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def cohort_catalogue_summary(framework, series: pd.DataFrame, name: str) -> list[dict[str, object]]:
    variant = framework.Variant("primary", "SPEI_{scale}_fixed", "SSI_{scale}_fixed")
    rows: list[dict[str, object]] = []
    for gage, group in series.groupby("GAGE_ID", sort=False):
        ordered = group.sort_values("date").reset_index(drop=True)
        for scale in (3, 6):
            resolved = framework.Variant(
                "primary",
                variant.met_column.format(scale=scale),
                variant.hyd_column.format(scale=scale),
                variant.onset_run,
                variant.onset_threshold,
                variant.termination_threshold,
                variant.termination_run,
                variant.window,
            )
            result, _ = framework.build_basin_catalogue(gage, scale, ordered, resolved)
            rows.extend(result)
    catalogue = pd.DataFrame(rows)
    summaries: list[dict[str, object]] = []
    for timescale, group in catalogue.groupby("timescale"):
        at_risk = group[~group.river_preexisting_at_met_onset]
        paired = at_risk[at_risk.assigned_hyd_onset.notna()]
        summaries.append({
            "cohort": name,
            "timescale": int(timescale),
            "n_input_basins": int(series.GAGE_ID.nunique()),
            "n_event_basins": int(group.GAGE_ID.nunique()),
            "n_eligible_events": len(group),
            "preexisting_percent": 100 * group.river_preexisting_at_met_onset.mean(),
            "active_at_termination_percent": 100 * group.river_active_at_met_termination.mean(),
            "assigned_new_onset_percent_of_at_risk": 100 * at_risk.assigned_hyd_onset.notna().mean(),
            "no_admissible_onset_percent_of_at_risk": 100 * at_risk.n_admissible_hydrologic_onsets.eq(0).mean(),
            "competition_percent_of_at_risk": 100 * at_risk.unassigned_despite_admissible_onset.mean(),
            "pair_direct_minus_assigned_active_percentage_points": 100 * (
                paired.river_active_at_met_termination.mean()
                - paired.assigned_termination_state.eq("assigned_event_active").mean()
            ),
        })
    return summaries


def outcome_summary(frame: pd.DataFrame, label: str) -> pd.DataFrame:
    complete = frame[frame.status.ne("incomplete_record")]
    result = (
        complete.groupby(["timescale", "status"])
        .size().rename("n_outcome").reset_index()
    )
    totals = complete.groupby("timescale").size()
    result["n_complete"] = result.timescale.map(totals)
    result["percent_of_complete"] = 100 * result.n_outcome / result.n_complete
    result.insert(0, "deduplication", label)
    return result


def main() -> None:
    framework = load_framework()
    series = pd.read_parquet(OUT / "audited_indices.parquet")
    series["GAGE_ID"] = series.GAGE_ID.astype(str).str.zfill(8)
    series["date"] = pd.to_datetime(series.date)
    primary = pd.read_parquet(OUT / "revised_state_catalogue_primary.parquet")

    print("Rebuilding ordered fixed-horizon recovery outcomes...", flush=True)
    recovery = framework.recovery_catalogue(primary, series[series.primary_population])
    recovery.to_parquet(OUT / "revised_recovery_event_catalogue.parquet", index=False, compression="zstd")
    full, conditional = framework.summarize_recovery(recovery)
    full.to_csv(OUT / "recovery_full_denominator_summary.csv", index=False)
    conditional.to_csv(OUT / "recovery_uninterrupted_conditional_summary.csv", index=False)

    print("Running population sensitivities...", flush=True)
    population = pd.read_csv(OUT / "audited_study_population.csv", dtype={"GAGE_ID": str})
    population["GAGE_ID"] = population.GAGE_ID.str.zfill(8)
    support_columns = [column for column in population if column.endswith("_min_calendar_support")]
    population["minimum_support"] = population[support_columns].min(axis=1)
    old = pd.read_csv(
        DATA_ROOT / "data" / "basins" / "legacy_3606_basin_ids.csv",
        usecols=["GAGE_ID"], dtype={"GAGE_ID": str},
    )
    old_ids = set(old.GAGE_ID.str.zfill(8))
    cohorts = {
        "calibration_support_10": set(population.loc[population.minimum_support.ge(10), "GAGE_ID"]),
        "calibration_support_25_primary": set(population.loc[population.minimum_support.ge(25), "GAGE_ID"]),
        "calibration_support_30": set(population.loc[population.minimum_support.ge(30), "GAGE_ID"]),
        "legacy_3606_intersection": old_ids & set(population.GAGE_ID),
    }
    cohort_rows: list[dict[str, object]] = []
    for name, ids in cohorts.items():
        print(f"  {name}: {len(ids):,} basins", flush=True)
        cohort_rows.extend(cohort_catalogue_summary(framework, series[series.GAGE_ID.isin(ids)], name))
    pd.DataFrame(cohort_rows).to_csv(OUT / "study_population_sensitivity.csv", index=False)

    print("Running unique-river-drought sensitivity...", flush=True)
    recovery12 = recovery[recovery.horizon_months.eq(12)].copy()
    key = ["GAGE_ID", "timescale", "actual_hyd_onset"]
    first_ids = (
        primary[primary.river_active_at_met_termination]
        .sort_values("met_termination").drop_duplicates(key, keep="first")
        [key + ["met_termination"]]
    )
    last_ids = (
        primary[primary.river_active_at_met_termination]
        .sort_values("met_termination").drop_duplicates(key, keep="last")
        [key + ["met_termination"]]
    )
    first = recovery12.merge(first_ids, on=key + ["met_termination"], how="inner")
    last = recovery12.merge(last_ids, on=key + ["met_termination"], how="inner")
    unique_summary = pd.concat([
        outcome_summary(recovery12, "all_boundary_encounters"),
        outcome_summary(first, "first_boundary_per_unique_river_drought"),
        outcome_summary(last, "last_boundary_per_unique_river_drought"),
    ], ignore_index=True)
    unique_summary.to_csv(OUT / "unique_river_drought_recovery_sensitivity.csv", index=False)

    first_primary = primary.merge(first_ids, on=key + ["met_termination"], how="inner")
    last_primary = primary.merge(last_ids, on=key + ["met_termination"], how="inner")
    persistence = []
    for label, frame in (
        ("all_boundary_encounters", primary),
        ("first_boundary_per_unique_river_drought", first_primary),
        ("last_boundary_per_unique_river_drought", last_primary),
    ):
        table = framework.operational_persistence(frame)
        table.insert(0, "deduplication", label)
        persistence.append(table)
    pd.concat(persistence, ignore_index=True).to_csv(
        OUT / "unique_river_drought_persistence_sensitivity.csv", index=False
    )

    attributes = framework.load_attributes()
    framework.stratification(primary, recovery, attributes).to_csv(
        OUT / "partially_pooled_hydrological_stratification.csv", index=False
    )

    audit = {
        "primary_population_basins": int(series.loc[series.primary_population, "GAGE_ID"].nunique()),
        "primary_event_basins": int(primary.GAGE_ID.nunique()),
        "active_boundary_encounters_3": int(primary.query("timescale == 3 and river_active_at_met_termination").shape[0]),
        "active_boundary_encounters_6": int(primary.query("timescale == 6 and river_active_at_met_termination").shape[0]),
        "unique_active_river_droughts_3": int(primary.query("timescale == 3 and river_active_at_met_termination").drop_duplicates(["GAGE_ID", "actual_hyd_onset"]).shape[0]),
        "unique_active_river_droughts_6": int(primary.query("timescale == 6 and river_active_at_met_termination").drop_duplicates(["GAGE_ID", "actual_hyd_onset"]).shape[0]),
    }
    (OUT / "cohort_and_unique_event_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    print("Cohort, recovery and unique-event sensitivities complete.", flush=True)


if __name__ == "__main__":
    main()
