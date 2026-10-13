#!/usr/bin/env python3
"""Test temporal stability in meteorological-boundary crossing.

The four prespecified outcomes are:

1. river active at meteorological termination (termination crossing);
2. river already active at meteorological onset (onset crossing);
3. either boundary crossed; and
4. the same river drought active at both boundaries (whole-event continuity).

The primary comparison is 1981--2002 versus 2003--2024. January--December
1980 is treated as burn-in for the start of the observed record. Sensitivities
exclude 2023--2024, add two further burn-in years, use only basins represented
in both periods, give each basin equal weight, and compare the fixed 1980--2010
calibration with the archived full-record indices.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm

from _paths import OUTPUTS, WORK


HERE = WORK / "temporal_change"
OUT = HERE / "outputs"
OUT.mkdir(parents=True, exist_ok=True)

SOURCE = Path(__file__).with_name("02_run_audited_state_framework.py")
INPUT = OUTPUTS / "audited_indices.parquet"
PRIMARY_CATALOGUE = OUTPUTS / "revised_state_catalogue_primary.parquet"

SEED = 7525
BOOTSTRAPS = 4000

METRICS = {
    "active_at_termination": "river_active_at_met_termination",
    "preexisting_at_onset": "river_preexisting_at_met_onset",
    "crosses_either_boundary": "crosses_either_boundary",
    "same_river_drought_spans_both_boundaries": "spans_both_boundaries",
}

SPLITS = {
    "full_45y_1980_2001_vs_2002_2024": ((1980, 2001), (2002, 2024)),
    "primary_1981_2002_vs_2003_2024": ((1981, 2002), (2003, 2024)),
    "onsets_through_2021_1981_2001_vs_2002_2021": ((1981, 2001), (2002, 2021)),
    "exclude_recent_1981_2001_vs_2002_2022": ((1981, 2001), (2002, 2022)),
    "extra_burnin_1983_2003_vs_2004_2024": ((1983, 2003), (2004, 2024)),
}

TREND_WINDOWS = {
    "primary_1981_2024": (1981, 2024),
    "onsets_through_2021_1981_2021": (1981, 2021),
    "exclude_recent_1981_2022": (1981, 2022),
    "extra_burnin_1983_2024": (1983, 2024),
    "both_edges_1983_2021": (1983, 2021),
    "both_edges_1983_2022": (1983, 2022),
}


def load_framework():
    spec = importlib.util.spec_from_file_location("state_framework", SOURCE)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import {SOURCE}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    module.EVAL_START = pd.Timestamp("1980-01-01")
    module.EVAL_END = pd.Timestamp("2024-12-01")
    return module


def build_catalogues(framework) -> pd.DataFrame:
    cache = OUT / "full_record_state_catalogue.parquet"
    if cache.exists():
        catalogue = pd.read_parquet(cache)
        for column in (
            "met_onset",
            "met_termination",
            "preexisting_hyd_onset",
            "preexisting_hyd_termination",
            "actual_hyd_onset",
            "actual_hyd_termination",
        ):
            catalogue[column] = pd.to_datetime(catalogue[column])
        return catalogue

    columns = [
        "GAGE_ID",
        "date",
        "SPEI_3",
        "SSI_3",
        "SPEI_6",
        "SSI_6",
        "SPEI_3_fixed",
        "SSI_3_fixed",
        "SPEI_6_fixed",
        "SSI_6_fixed",
    ]
    series = pd.read_parquet(
        INPUT,
        columns=columns + ["primary_population"],
    )
    series = series[series.primary_population].copy()
    series["GAGE_ID"] = series.GAGE_ID.astype(str).str.zfill(8)
    series["date"] = pd.to_datetime(series.date)
    series.sort_values(["GAGE_ID", "date"], inplace=True)

    variants = [
        framework.Variant(
            "fixed_1980_2010", "SPEI_{scale}_fixed", "SSI_{scale}_fixed"
        ),
        framework.Variant(
            "full_record_calibration", "SPEI_{scale}", "SSI_{scale}"
        ),
    ]
    rows: list[dict[str, object]] = []
    audits: list[dict[str, object]] = []
    grouped = series.groupby("GAGE_ID", sort=False)
    for number, (gage, group) in enumerate(grouped, start=1):
        for variant in variants:
            for scale in (3, 6):
                resolved = framework.Variant(
                    variant.name,
                    variant.met_column.format(scale=scale),
                    variant.hyd_column.format(scale=scale),
                    onset_run=variant.onset_run,
                    onset_threshold=variant.onset_threshold,
                    termination_threshold=variant.termination_threshold,
                    termination_run=variant.termination_run,
                    window=variant.window,
                )
                catalogue_rows, audit_rows = framework.build_basin_catalogue(
                    gage, scale, group, resolved
                )
                rows.extend(catalogue_rows)
                audits.extend(audit_rows)
        if number % 500 == 0:
            print(f"catalogue: {number:,} basins", flush=True)

    catalogue = pd.DataFrame(rows)
    audit = pd.DataFrame(audits)
    for column in (
        "met_onset",
        "met_termination",
        "preexisting_hyd_onset",
        "preexisting_hyd_termination",
        "actual_hyd_onset",
        "actual_hyd_termination",
    ):
        catalogue[column] = pd.to_datetime(catalogue[column])

    catalogue["crosses_either_boundary"] = (
        catalogue.river_preexisting_at_met_onset
        | catalogue.river_active_at_met_termination
    )
    catalogue["spans_both_boundaries"] = (
        catalogue.river_preexisting_at_met_onset
        & catalogue.river_active_at_met_termination
        & catalogue.preexisting_hyd_onset.notna()
        & (catalogue.preexisting_hyd_onset == catalogue.actual_hyd_onset)
    )
    catalogue["onset_year"] = catalogue.met_onset.dt.year.astype(int)
    catalogue["onset_month"] = catalogue.met_onset.dt.month.astype(int)

    catalogue.to_parquet(OUT / "full_record_state_catalogue.parquet", index=False)
    audit.to_parquet(OUT / "full_record_eligibility_audit.parquet", index=False)
    return catalogue


def validate_against_primary(catalogue: pd.DataFrame) -> dict[str, object]:
    primary = pd.read_parquet(PRIMARY_CATALOGUE)
    primary["GAGE_ID"] = primary.GAGE_ID.astype(str).str.zfill(8)
    primary["met_onset"] = pd.to_datetime(primary.met_onset)
    rebuilt = catalogue.loc[
        (catalogue.variant == "fixed_1980_2010")
        & catalogue.met_onset.between("2011-01-01", "2024-12-01")
    ].copy()
    keys = ["GAGE_ID", "timescale", "met_onset"]
    cols = [
        *keys,
        "river_preexisting_at_met_onset",
        "river_active_at_met_termination",
        "termination_state",
    ]
    check = primary[cols].merge(
        rebuilt[cols], on=keys, how="outer", suffixes=("_primary", "_rebuilt"), indicator=True
    )
    shared = check._merge.eq("both")
    outcome_match = (
        check.loc[shared, "river_preexisting_at_met_onset_primary"].astype(bool).to_numpy()
        == check.loc[shared, "river_preexisting_at_met_onset_rebuilt"].astype(bool).to_numpy()
    ) & (
        check.loc[shared, "river_active_at_met_termination_primary"].astype(bool).to_numpy()
        == check.loc[shared, "river_active_at_met_termination_rebuilt"].astype(bool).to_numpy()
    ) & (
        check.loc[shared, "termination_state_primary"].astype(str).to_numpy()
        == check.loc[shared, "termination_state_rebuilt"].astype(str).to_numpy()
    )
    return {
        "primary_rows": int(len(primary)),
        "rebuilt_2011_2024_rows": int(len(rebuilt)),
        "shared_rows": int(shared.sum()),
        "primary_only_rows": int((check._merge == "left_only").sum()),
        "rebuilt_only_rows": int((check._merge == "right_only").sum()),
        "direct_state_outcomes_matching_shared_rows": int(outcome_match.sum()),
        "direct_state_outcomes_mismatching_shared_rows": int((~outcome_match).sum()),
    }


def add_period(group: pd.DataFrame, early: tuple[int, int], late: tuple[int, int]) -> pd.DataFrame:
    selected = group.loc[
        group.onset_year.between(early[0], early[1])
        | group.onset_year.between(late[0], late[1])
    ].copy()
    selected["period"] = np.where(
        selected.onset_year.between(early[0], early[1]), "early", "late"
    )
    return selected


def cluster_bootstrap_split(
    selected: pd.DataFrame,
    outcome: str,
    weighting: str,
    rng: np.random.Generator,
) -> dict[str, float | int]:
    aggregate = (
        selected.groupby(["GAGE_ID", "period"], observed=True)[outcome]
        .agg(successes="sum", events="size")
        .reset_index()
    )
    successes = aggregate.pivot(index="GAGE_ID", columns="period", values="successes").fillna(0)
    events = aggregate.pivot(index="GAGE_ID", columns="period", values="events").fillna(0)
    for period in ("early", "late"):
        if period not in successes:
            successes[period] = 0.0
            events[period] = 0.0
    successes = successes[["early", "late"]]
    events = events[["early", "late"]]

    if weighting in {"balanced_event_pooled", "balanced_equal_basin"}:
        keep = (events.early > 0) & (events.late > 0)
        successes = successes.loc[keep]
        events = events.loc[keep]

    s = successes.to_numpy(dtype=float)
    n = events.to_numpy(dtype=float)
    basin_count = len(s)
    if basin_count == 0:
        raise RuntimeError("No basins available for split estimate")

    if weighting in {"all_event_pooled", "balanced_event_pooled"}:
        observed = s.sum(axis=0) / n.sum(axis=0)

        def statistic(indices: np.ndarray) -> np.ndarray:
            return s[indices].sum(axis=1) / n[indices].sum(axis=1)

    elif weighting == "balanced_equal_basin":
        rates = s / n
        observed = rates.mean(axis=0)

        def statistic(indices: np.ndarray) -> np.ndarray:
            return rates[indices].mean(axis=1)

    else:
        raise ValueError(weighting)

    differences = np.empty(BOOTSTRAPS, dtype=float)
    batch = 250
    for start in range(0, BOOTSTRAPS, batch):
        stop = min(start + batch, BOOTSTRAPS)
        indices = rng.integers(0, basin_count, size=(stop - start, basin_count))
        boot_rates = statistic(indices)
        differences[start:stop] = boot_rates[:, 1] - boot_rates[:, 0]

    difference = float(observed[1] - observed[0])
    lower, upper = np.quantile(differences, [0.025, 0.975])
    p_boot = 2 * min(
        (np.count_nonzero(differences <= 0) + 1) / (BOOTSTRAPS + 1),
        (np.count_nonzero(differences >= 0) + 1) / (BOOTSTRAPS + 1),
    )
    return {
        "n_basins": basin_count,
        "n_early": int(n[:, 0].sum()),
        "n_late": int(n[:, 1].sum()),
        "early_percent": 100 * float(observed[0]),
        "late_percent": 100 * float(observed[1]),
        "difference_pp": 100 * difference,
        "ci_low_pp": 100 * float(lower),
        "ci_high_pp": 100 * float(upper),
        "bootstrap_p_two_sided": min(1.0, float(p_boot)),
    }


def split_results(catalogue: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    rng = np.random.default_rng(SEED)
    for variant in ("fixed_1980_2010", "full_record_calibration"):
        for scale in (3, 6):
            base = catalogue.loc[
                (catalogue.variant == variant) & (catalogue.timescale == scale)
            ]
            for split_name, (early, late) in SPLITS.items():
                selected = add_period(base, early, late)
                for metric_name, outcome in METRICS.items():
                    for weighting in (
                        "all_event_pooled",
                        "balanced_event_pooled",
                        "balanced_equal_basin",
                    ):
                        estimate = cluster_bootstrap_split(
                            selected, outcome, weighting, rng
                        )
                        rows.append(
                            {
                                "variant": variant,
                                "timescale": scale,
                                "split": split_name,
                                "early_years": f"{early[0]}-{early[1]}",
                                "late_years": f"{late[0]}-{late[1]}",
                                "metric": metric_name,
                                "weighting": weighting,
                                **estimate,
                            }
                        )
    result = pd.DataFrame(rows)
    result.to_csv(OUT / "period_split_estimates.csv", index=False)
    return result


def alternating_residualize(
    values: np.ndarray, groups: list[np.ndarray], tolerance: float = 1e-12
) -> np.ndarray:
    residual = np.asarray(values, dtype=float).copy()
    for _ in range(500):
        before = residual.copy()
        for codes in groups:
            count = np.bincount(codes)
            means = np.bincount(codes, weights=residual) / count
            residual -= means[codes]
        if np.max(np.abs(residual - before)) < tolerance:
            break
    return residual


def cluster_meat(scores: np.ndarray, codes: np.ndarray) -> float:
    totals = np.bincount(codes, weights=scores)
    group_count = len(totals)
    correction = group_count / (group_count - 1) if group_count > 1 else 1.0
    return correction * float(np.square(totals).sum())


def fixed_effect_trend(selected: pd.DataFrame, outcome: str) -> dict[str, float | int]:
    data = selected[["GAGE_ID", "onset_year", "onset_month", outcome]].copy()
    basin_codes, basin_levels = pd.factorize(data.GAGE_ID, sort=True)
    month_codes, _ = pd.factorize(data.onset_month, sort=True)
    year_codes, year_levels = pd.factorize(data.onset_year, sort=True)
    basin_year_codes, basin_year_levels = pd.factorize(
        pd.MultiIndex.from_arrays([basin_codes, year_codes]), sort=True
    )

    x = (data.onset_year.to_numpy(dtype=float) - data.onset_year.mean()) / 10.0
    y = data[outcome].to_numpy(dtype=float)
    x_resid = alternating_residualize(x, [basin_codes, month_codes])
    y_resid = alternating_residualize(y, [basin_codes, month_codes])
    xx = float(x_resid @ x_resid)
    beta = float(x_resid @ y_resid / xx)
    errors = y_resid - beta * x_resid
    scores = x_resid * errors
    variance = (
        cluster_meat(scores, basin_codes)
        + cluster_meat(scores, year_codes)
        - cluster_meat(scores, basin_year_codes)
    ) / (xx * xx)
    variance = max(0.0, variance)
    se = float(np.sqrt(variance))
    degrees = max(1, min(len(basin_levels), len(year_levels)) - 1)
    critical = float(stats.t.ppf(0.975, degrees))
    p_value = (
        float(2 * stats.t.sf(abs(beta / se), degrees)) if se > 0 else np.nan
    )
    return {
        "n_events": len(data),
        "n_basins": len(basin_levels),
        "n_years": len(year_levels),
        "slope_pp_per_decade": 100 * beta,
        "se_pp_per_decade": 100 * se,
        "ci_low_pp_per_decade": 100 * (beta - critical * se),
        "ci_high_pp_per_decade": 100 * (beta + critical * se),
        "p_two_sided": p_value,
    }


def trend_results(catalogue: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for variant in ("fixed_1980_2010", "full_record_calibration"):
        for scale in (3, 6):
            base = catalogue.loc[
                (catalogue.variant == variant) & (catalogue.timescale == scale)
            ]
            for window_name, (start, end) in TREND_WINDOWS.items():
                selected = base.loc[base.onset_year.between(start, end)]
                for metric_name, outcome in METRICS.items():
                    estimate = fixed_effect_trend(selected, outcome)
                    rows.append(
                        {
                            "variant": variant,
                            "timescale": scale,
                            "window": window_name,
                            "start_year": start,
                            "end_year": end,
                            "metric": metric_name,
                            **estimate,
                        }
                    )
    result = pd.DataFrame(rows)
    result.to_csv(OUT / "within_basin_trend_estimates.csv", index=False)
    return result


def annual_results(catalogue: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (variant, scale, year), group in catalogue.groupby(
        ["variant", "timescale", "onset_year"], observed=True
    ):
        for metric_name, outcome in METRICS.items():
            rows.append(
                {
                    "variant": variant,
                    "timescale": scale,
                    "onset_year": year,
                    "metric": metric_name,
                    "n_events": len(group),
                    "n_outcome": int(group[outcome].sum()),
                    "percent": 100 * float(group[outcome].mean()),
                    "n_basins_with_events": int(group.GAGE_ID.nunique()),
                }
            )
    annual = pd.DataFrame(rows)
    annual.to_csv(OUT / "annual_boundary_crossing_shares.csv", index=False)

    hac_rows: list[dict[str, object]] = []
    for (variant, scale, metric), group in annual.loc[
        annual.onset_year.between(1981, 2024)
    ].groupby(["variant", "timescale", "metric"], observed=True):
        x = (group.onset_year.to_numpy(dtype=float) - group.onset_year.mean()) / 10.0
        model = sm.WLS(
            group.percent.to_numpy(dtype=float),
            sm.add_constant(x),
            weights=group.n_events.to_numpy(dtype=float),
        ).fit(cov_type="HAC", cov_kwds={"maxlags": 3, "use_correction": True})
        hac_rows.append(
            {
                "variant": variant,
                "timescale": scale,
                "metric": metric,
                "n_years": len(group),
                "slope_pp_per_decade": float(model.params[1]),
                "se_pp_per_decade": float(model.bse[1]),
                "ci_low_pp_per_decade": float(model.conf_int()[1, 0]),
                "ci_high_pp_per_decade": float(model.conf_int()[1, 1]),
                "p_two_sided": float(model.pvalues[1]),
            }
        )
    pd.DataFrame(hac_rows).to_csv(OUT / "annual_hac_trend_estimates.csv", index=False)
    return annual


def temporal_split_results(catalogue: pd.DataFrame) -> pd.DataFrame:
    """Infer period contrasts across annual climate realizations.

    The basin-cluster bootstrap in period_split_estimates.csv addresses spatial
    dependence. This complementary analysis treats onset year as the sampling
    unit and uses a three-year Newey--West correction for serial dependence.
    """
    rows: list[dict[str, object]] = []
    for variant in ("fixed_1980_2010", "full_record_calibration"):
        for scale in (3, 6):
            base = catalogue.loc[
                (catalogue.variant == variant) & (catalogue.timescale == scale)
            ]
            for split_name, (early, late) in SPLITS.items():
                selected = add_period(base, early, late)
                for sample in ("all_basins", "balanced_basins"):
                    working = selected
                    if sample == "balanced_basins":
                        coverage = (
                            selected.groupby(["GAGE_ID", "period"], observed=True)
                            .size()
                            .unstack(fill_value=0)
                        )
                        if "early" not in coverage:
                            coverage["early"] = 0
                        if "late" not in coverage:
                            coverage["late"] = 0
                        ids = coverage.index[
                            (coverage.early > 0) & (coverage.late > 0)
                        ]
                        working = selected.loc[selected.GAGE_ID.isin(ids)]
                    for metric_name, outcome in METRICS.items():
                        annual = (
                            working.groupby(["onset_year", "period"], observed=True)[outcome]
                            .agg(successes="sum", events="size")
                            .reset_index()
                            .sort_values("onset_year")
                        )
                        annual["share"] = annual.successes / annual.events
                        annual["late"] = (annual.period == "late").astype(int)
                        model = sm.WLS(
                            annual.share.to_numpy(dtype=float),
                            sm.add_constant(annual.late.to_numpy(dtype=float)),
                            weights=annual.events.to_numpy(dtype=float),
                        ).fit(
                            cov_type="HAC",
                            cov_kwds={"maxlags": 3, "use_correction": True},
                            use_t=True,
                        )
                        interval = model.conf_int()[1]
                        rows.append(
                            {
                                "variant": variant,
                                "timescale": scale,
                                "split": split_name,
                                "sample": sample,
                                "metric": metric_name,
                                "n_years": len(annual),
                                "n_events": int(annual.events.sum()),
                                "difference_pp": 100 * float(model.params[1]),
                                "se_pp": 100 * float(model.bse[1]),
                                "ci_low_pp": 100 * float(interval[0]),
                                "ci_high_pp": 100 * float(interval[1]),
                                "p_two_sided": float(model.pvalues[1]),
                            }
                        )
    result = pd.DataFrame(rows)
    result.to_csv(OUT / "period_split_temporal_hac_estimates.csv", index=False)
    return result


def multidecadal_results(catalogue: pd.DataFrame) -> pd.DataFrame:
    bins = [1980, 1991, 2002, 2013, 2024]
    labels = ["1981-1991", "1992-2002", "2003-2013", "2014-2024"]
    selected = catalogue.loc[catalogue.onset_year.between(1981, 2024)].copy()
    selected["period"] = pd.cut(selected.onset_year, bins=bins, labels=labels)
    rows: list[dict[str, object]] = []
    for (variant, scale, period), group in selected.groupby(
        ["variant", "timescale", "period"], observed=True
    ):
        for metric_name, outcome in METRICS.items():
            rows.append(
                {
                    "variant": variant,
                    "timescale": scale,
                    "period": str(period),
                    "metric": metric_name,
                    "n_events": len(group),
                    "percent": 100 * float(group[outcome].mean()),
                    "mean_met_duration_months": float(group.met_duration_months.mean()),
                }
            )
    result = pd.DataFrame(rows)
    result.to_csv(OUT / "eleven_year_period_shares.csv", index=False)
    return result


def write_summary(
    catalogue: pd.DataFrame,
    validation: dict[str, object],
    splits: pd.DataFrame,
    trends: pd.DataFrame,
) -> None:
    selector = (
        (splits.variant == "fixed_1980_2010")
        & (splits.split == "primary_1981_2002_vs_2003_2024")
        & (splits.metric == "active_at_termination")
        & (splits.weighting == "all_event_pooled")
    )
    primary_split = splits.loc[selector].sort_values("timescale")
    selector = (
        (trends.variant == "fixed_1980_2010")
        & (trends.window == "primary_1981_2024")
        & (trends.metric == "active_at_termination")
    )
    primary_trend = trends.loc[selector].sort_values("timescale")
    payload = {
        "analysis_scope": {
            "catalogue_rows": int(len(catalogue)),
            "basins": int(catalogue.GAGE_ID.nunique()),
            "onset_start": str(catalogue.met_onset.min().date()),
            "onset_end": str(catalogue.met_onset.max().date()),
            "bootstrap_resamples": BOOTSTRAPS,
        },
        "validation": validation,
        "primary_termination_crossing_split": primary_split.to_dict("records"),
        "primary_termination_crossing_trend": primary_trend.to_dict("records"),
    }
    (OUT / "diagnostic_summary.json").write_text(
        json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8"
    )


def main() -> None:
    framework = load_framework()
    catalogue = build_catalogues(framework)
    validation = validate_against_primary(catalogue)
    splits = split_results(catalogue)
    trends = trend_results(catalogue)
    annual_results(catalogue)
    temporal_split_results(catalogue)
    multidecadal_results(catalogue)
    write_summary(catalogue, validation, splits, trends)
    print(json.dumps(validation, indent=2), flush=True)
    print("\nPrimary termination-crossing split", flush=True)
    print(
        splits.loc[
            (splits.variant == "fixed_1980_2010")
            & (splits.split == "primary_1981_2002_vs_2003_2024")
            & (splits.metric == "active_at_termination")
            & (splits.weighting == "all_event_pooled"),
            [
                "timescale",
                "n_early",
                "n_late",
                "early_percent",
                "late_percent",
                "difference_pp",
                "ci_low_pp",
                "ci_high_pp",
                "bootstrap_p_two_sided",
            ],
        ].to_string(index=False),
        flush=True,
    )


if __name__ == "__main__":
    main()
