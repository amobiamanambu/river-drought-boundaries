#!/usr/bin/env python3
"""Test the incremental held-out information in NLDAS root-zone moisture.

Primary predictors are restricted to information available before
meteorological onset. Separate retrospective information sets are reported for
comparison. Models with and without NLDAS use identical events and folds.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score, roc_auc_score
from sklearn.model_selection import GroupKFold

from _paths import BASIN_ATTRIBUTES, NLDAS_MONTHLY, OUTPUTS

OUT = OUTPUTS


SEED = 7525
BOOTSTRAPS = 500
TEMPORAL_CUT = 2017
NLDAS_FEATURE = "RootMoist_z_pre1"
STATIC_FEATURES = [
    "RUNAVE7100",
    "PPTAVG_BASIN",
    "T_AVG_BASIN",
    "RH_BASIN",
    "PET",
    "SNOW_PCT_PRECIP",
    "PRECIP_SEAS_IND",
    "DRAIN_SQKM",
    "SLOPE_PCT",
    "AWCAVE",
    "PERMAVE",
    "ROCKDEPAVE",
    "WTDEPAVE",
    "CONTACT",
]


def month_start(values: pd.Series) -> pd.Series:
    return pd.to_datetime(values).dt.to_period("M").dt.to_timestamp()


def add_season(frame: pd.DataFrame) -> None:
    month = frame.met_onset.dt.month
    frame["onset_sin"] = np.sin(2 * np.pi * month / 12)
    frame["onset_cos"] = np.cos(2 * np.pi * month / 12)


def load_event_features() -> pd.DataFrame:
    events = pd.read_parquet(OUT / "revised_state_catalogue_primary.parquet")
    events["GAGE_ID"] = events.GAGE_ID.astype(str).str.zfill(8)
    events["met_onset"] = month_start(events.met_onset)
    events["met_termination"] = month_start(events.met_termination)
    events["onset_year"] = events.met_onset.dt.year
    add_season(events)

    static = pd.read_csv(BASIN_ATTRIBUTES, dtype={"GAGE_ID": str, "HUC02": str})
    static["GAGE_ID"] = static.GAGE_ID.astype(str).str.zfill(8)
    events = events.merge(
        static[["GAGE_ID", "HUC02", *[c for c in STATIC_FEATURES if c in static]]],
        on="GAGE_ID",
        how="left",
        validate="many_to_one",
    )

    indices = pd.read_parquet(
        OUT / "audited_indices.parquet",
        columns=[
            "GAGE_ID",
            "date",
            "SPEI_3_fixed",
            "SPEI_6_fixed",
            "SSI_3_fixed",
            "SSI_6_fixed",
        ],
    )
    indices["GAGE_ID"] = indices.GAGE_ID.astype(str).str.zfill(8)
    indices["date"] = month_start(indices.date)
    indices.sort_values(["GAGE_ID", "date"], inplace=True)

    pieces: list[pd.DataFrame] = []
    for timescale in (3, 6):
        spei = f"SPEI_{timescale}_fixed"
        ssi = f"SSI_{timescale}_fixed"
        lookup = indices[["GAGE_ID", "date", spei, ssi]].copy()
        lookup["spei_pre3_mean"] = lookup.groupby("GAGE_ID", sort=False)[spei].transform(
            lambda values: values.shift(1).rolling(3, min_periods=3).mean()
        )
        lookup["ssi_pre1"] = lookup.groupby("GAGE_ID", sort=False)[ssi].shift(1)
        lookup["ssi_pre3_mean"] = lookup.groupby("GAGE_ID", sort=False)[ssi].transform(
            lambda values: values.shift(1).rolling(3, min_periods=3).mean()
        )
        lookup.rename(
            columns={
                "date": "met_onset",
                spei: "spei_at_onset",
                ssi: "ssi_at_onset",
            },
            inplace=True,
        )
        piece = events[events.timescale.eq(timescale)].merge(
            lookup,
            on=["GAGE_ID", "met_onset"],
            how="left",
            validate="many_to_one",
        )
        pieces.append(piece)
    events = pd.concat(pieces, ignore_index=True)

    storage = pd.read_parquet(
        NLDAS_MONTHLY,
        columns=["GAGE_ID", "time", "RootMoist"],
    )
    storage["GAGE_ID"] = storage.GAGE_ID.astype(str).str.zfill(8)
    storage["time"] = month_start(storage.time)
    storage["calendar_month"] = storage.time.dt.month
    calibration = storage[storage.time.between("1981-01-01", "2010-12-01")]
    climatology = (
        calibration.groupby(["GAGE_ID", "calendar_month"]).RootMoist
        .agg(["mean", "std"])
        .reset_index()
    )
    storage = storage.merge(
        climatology,
        on=["GAGE_ID", "calendar_month"],
        how="left",
        validate="many_to_one",
    )
    storage["RootMoist_z"] = (storage.RootMoist - storage["mean"]) / storage["std"]
    storage.sort_values(["GAGE_ID", "time"], inplace=True)
    storage[NLDAS_FEATURE] = storage.groupby("GAGE_ID", sort=False).RootMoist_z.shift(1)
    nldas = storage[["GAGE_ID", "time", NLDAS_FEATURE]].rename(columns={"time": "met_onset"})
    return events.merge(
        nldas,
        on=["GAGE_ID", "met_onset"],
        how="left",
        validate="many_to_one",
    )


def select_features(frame: pd.DataFrame, requested: list[str]) -> list[str]:
    selected: list[str] = []
    for feature in requested:
        if feature not in frame:
            continue
        values = pd.to_numeric(frame[feature], errors="coerce")
        if values.notna().mean() >= 0.80 and values.nunique(dropna=True) > 1:
            selected.append(feature)
    return selected


def make_splits(frame: pd.DataFrame, scheme: str) -> list[tuple[np.ndarray, np.ndarray]]:
    huc = frame.HUC02.astype(str).to_numpy()
    years = frame.onset_year.to_numpy()
    if scheme == "spatial_huc02":
        return list(GroupKFold(n_splits=5).split(frame, groups=huc))
    if scheme == "temporal":
        return [
            (
                np.flatnonzero(years <= TEMPORAL_CUT),
                np.flatnonzero(years > TEMPORAL_CUT),
            )
        ]
    if scheme == "joint_space_time":
        result: list[tuple[np.ndarray, np.ndarray]] = []
        for raw_train, raw_test in GroupKFold(n_splits=5).split(frame, groups=huc):
            train = raw_train[years[raw_train] <= TEMPORAL_CUT]
            test = raw_test[years[raw_test] > TEMPORAL_CUT]
            result.append((train, test))
        return result
    raise ValueError(scheme)


def predict(
    frame: pd.DataFrame,
    target: np.ndarray,
    features: list[str],
    scheme: str,
    task: str,
) -> np.ndarray:
    X = frame[features].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    prediction = np.full(len(frame), np.nan)
    for train, test in make_splits(frame, scheme):
        imputer = SimpleImputer(strategy="median")
        X_train = imputer.fit_transform(X[train])
        X_test = imputer.transform(X[test])
        if task == "classification":
            model = ExtraTreesClassifier(
                n_estimators=160,
                max_depth=10,
                min_samples_leaf=20,
                max_features=0.8,
                class_weight="balanced",
                n_jobs=4,
                random_state=SEED,
            )
            model.fit(X_train, target[train].astype(int))
            prediction[test] = model.predict_proba(X_test)[:, 1]
        else:
            model = ExtraTreesRegressor(
                n_estimators=160,
                max_depth=10,
                min_samples_leaf=20,
                max_features=0.8,
                n_jobs=4,
                random_state=SEED,
            )
            model.fit(X_train, target[train].astype(float))
            prediction[test] = model.predict(X_test)
    return prediction


def score(target: np.ndarray, prediction: np.ndarray, task: str) -> float:
    valid = np.isfinite(target) & np.isfinite(prediction)
    if valid.sum() == 0:
        return np.nan
    if task == "classification":
        if np.unique(target[valid]).size < 2:
            return np.nan
        return float(roc_auc_score(target[valid], prediction[valid]))
    return float(r2_score(target[valid], prediction[valid]))


def basin_bootstrap_delta(
    frame: pd.DataFrame,
    target: np.ndarray,
    baseline: np.ndarray,
    augmented: np.ndarray,
    task: str,
) -> tuple[float, float, int]:
    valid = np.isfinite(target) & np.isfinite(baseline) & np.isfinite(augmented)
    target = target[valid]
    baseline = baseline[valid]
    augmented = augmented[valid]
    basin = frame.GAGE_ID.to_numpy()[valid]
    unique = np.unique(basin)
    locations = {name: np.flatnonzero(basin == name) for name in unique}
    rng = np.random.default_rng(SEED)
    deltas: list[float] = []
    for _ in range(BOOTSTRAPS):
        sampled = rng.choice(unique, size=len(unique), replace=True)
        index = np.concatenate([locations[name] for name in sampled])
        base_score = score(target[index], baseline[index], task)
        augmented_score = score(target[index], augmented[index], task)
        if np.isfinite(base_score) and np.isfinite(augmented_score):
            deltas.append(augmented_score - base_score)
    low, high = np.quantile(deltas, [0.025, 0.975])
    return float(low), float(high), len(deltas)


def run_comparisons(events: pd.DataFrame) -> pd.DataFrame:
    strict = STATIC_FEATURES + [
        "spei_pre3_mean",
        "ssi_pre1",
        "ssi_pre3_mean",
        "onset_sin",
        "onset_cos",
    ]
    retrospective_no_streamflow = STATIC_FEATURES + [
        "spei_at_onset",
        "spei_pre3_mean",
        "onset_sin",
        "onset_cos",
    ]
    retrospective_with_streamflow = retrospective_no_streamflow + [
        "ssi_pre1",
        "ssi_pre3_mean",
        "ssi_at_onset",
    ]
    information_sets = {
        "strict_pre_onset": strict,
        "retrospective_without_streamflow_state": retrospective_no_streamflow,
        "retrospective_with_contemporaneous_ssi": retrospective_with_streamflow,
    }

    rows: list[dict[str, object]] = []
    for timescale in (3, 6):
        scale = events[events.timescale.eq(timescale)].copy()
        at_risk = scale[~scale.river_preexisting_at_met_onset].copy()
        propagated = at_risk[
            at_risk.onset_pathway.ne("no_new_river_onset_within_window")
        ].copy()
        tasks = {
            "new_onset_within_window": (
                at_risk,
                at_risk.onset_pathway.ne("no_new_river_onset_within_window").astype(float).to_numpy(),
                "classification",
            ),
            "new_onset_by_confirmation_given_new_onset": (
                propagated,
                propagated.assigned_onset_lag_months.le(2).astype(float).to_numpy(),
                "classification",
            ),
            "onset_lag_months_given_new_onset": (
                propagated,
                propagated.assigned_onset_lag_months.astype(float).to_numpy(),
                "regression",
            ),
        }
        for outcome, (raw, target_raw, task) in tasks.items():
            raw = raw.reset_index(drop=True)
            target_raw = np.asarray(target_raw, dtype=float)
            for information_set, requested_features in information_sets.items():
                base_features = select_features(raw, requested_features)
                required = base_features + [NLDAS_FEATURE, "HUC02", "GAGE_ID", "onset_year"]
                valid = raw[required].notna().all(axis=1).to_numpy() & np.isfinite(target_raw)
                frame = raw.loc[valid].reset_index(drop=True)
                target = target_raw[valid]
                if len(frame) < 300 or frame.HUC02.nunique() < 5:
                    continue
                for scheme in ("spatial_huc02", "temporal", "joint_space_time"):
                    baseline = predict(frame, target, base_features, scheme, task)
                    augmented = predict(
                        frame,
                        target,
                        base_features + [NLDAS_FEATURE],
                        scheme,
                        task,
                    )
                    baseline_score = score(target, baseline, task)
                    augmented_score = score(target, augmented, task)
                    low, high, n_boot = basin_bootstrap_delta(
                        frame, target, baseline, augmented, task
                    )
                    rows.append(
                        {
                            "timescale": timescale,
                            "outcome": outcome,
                            "task": task,
                            "information_set": information_set,
                            "validation": scheme,
                            "n_events_total": len(frame),
                            "n_events_scored": int(
                                (np.isfinite(baseline) & np.isfinite(augmented)).sum()
                            ),
                            "n_basins": int(frame.GAGE_ID.nunique()),
                            "baseline_score": baseline_score,
                            "with_nldas_score": augmented_score,
                            "delta_score": augmented_score - baseline_score,
                            "delta_score_percentage_points": 100
                            * (augmented_score - baseline_score),
                            "delta_ci_low": low,
                            "delta_ci_high": high,
                            "delta_ci_low_percentage_points": 100 * low,
                            "delta_ci_high_percentage_points": 100 * high,
                            "n_bootstrap": n_boot,
                            "metric": "AUC" if task == "classification" else "R2",
                            "baseline_features": ";".join(base_features),
                            "added_feature": NLDAS_FEATURE,
                        }
                    )
    return pd.DataFrame(rows)


def audit_results(results: pd.DataFrame) -> dict[str, object]:
    return {
        "n_rows": int(len(results)),
        "expected_rows": 54,
        "all_comparisons_have_500_bootstraps": bool(results.n_bootstrap.eq(BOOTSTRAPS).all()),
        "all_scores_finite": bool(
            np.isfinite(results[["baseline_score", "with_nldas_score", "delta_score"]]).all().all()
        ),
        "same_cases_within_each_comparison": True,
        "strict_information_cutoff": "All dynamic predictors end in month -1 relative to meteorological onset.",
        "retrospective_warning": "The other two information sets use meteorological or streamflow state at the retrospectively identified onset month and are not prospective forecasts.",
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    print("Loading revised full-denominator catalogue and origin-timed covariates...", flush=True)
    events = load_event_features()
    print(f"Loaded {len(events):,} eligible meteorological events.", flush=True)
    results = run_comparisons(events)
    output = OUT / "nldas_strict_timing_comparison.csv"
    results.to_csv(output, index=False)
    audit = audit_results(results)
    (OUT / "nldas_strict_timing_audit.json").write_text(
        json.dumps(audit, indent=2) + "\n"
    )
    print(json.dumps(audit, indent=2), flush=True)
    print("\nStrict pre-onset comparisons:", flush=True)
    print(
        results[results.information_set.eq("strict_pre_onset")][
            [
                "timescale",
                "outcome",
                "validation",
                "n_events_scored",
                "metric",
                "baseline_score",
                "with_nldas_score",
                "delta_score_percentage_points",
                "delta_ci_low_percentage_points",
                "delta_ci_high_percentage_points",
            ]
        ].to_string(index=False),
        flush=True,
    )
    print(f"\nSaved {output}", flush=True)


if __name__ == "__main__":
    main()
