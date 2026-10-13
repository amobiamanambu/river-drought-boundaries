#!/usr/bin/env python3
"""Reconstruct the study population and drought indices from source data.

The script rebuilds daily-discharge quality control, monthly inputs and
fixed/full-period drought indices from the public upstream observations.

Primary population requirements
-------------------------------
* conterminous GAGES-II gauge with a basin inventory record;
* at least 30 years between first and last valid daily discharge observation;
* at least 90% valid daily discharge within that observed span; and
* at least 25 valid accumulated calibration values for every calendar month
  used by SPEI-3/6 and SSI-1/3/6 during 1980--2010.

The last requirement makes the fixed-period distribution support explicit.
The broader daily-QC population is also retained for sensitivity analysis.
"""

from __future__ import annotations

import argparse
import gzip
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import special, stats

from _paths import OUTPUTS, PROJECT_ROOT


ROOT = PROJECT_ROOT
OUT = OUTPUTS
OUT.mkdir(parents=True, exist_ok=True)

START = pd.Timestamp("1980-01-01")
END = pd.Timestamp("2025-12-31")
CAL_END = pd.Timestamp("2010-12-31")
MIN_RECORD_YEARS = 30.0
MIN_DAILY_COMPLETENESS = 0.90
MIN_CALENDAR_MONTH_SUPPORT = 25


def conterminous_gages() -> set[str]:
    path = ROOT / "gagesii_csv" / "conterm_basinid.txt"
    frame = pd.read_csv(path, dtype={"STAID": str}, encoding="latin-1")
    return set(frame.STAID.astype(str).str.strip().str.zfill(8))


def inventory() -> pd.DataFrame:
    frame = pd.read_csv(ROOT / "basin_inventory.csv", dtype={"GAGE_ID": str})
    frame["GAGE_ID"] = frame.GAGE_ID.astype(str).str.zfill(8)
    frame = frame[frame.GAGE_ID.isin(conterminous_gages())].drop_duplicates("GAGE_ID")
    return frame


def read_rdb_sections(path: Path):
    """Yield one vectorized daily frame per gauge section in a USGS RDB file."""
    with gzip.open(path, "rt", errors="replace") as stream:
        lines = [line.rstrip("\n") for line in stream if not line.startswith("#")]
    starts = [index for index, line in enumerate(lines) if line.startswith("agency_cd\t")]
    starts.append(len(lines))
    for position in range(len(starts) - 1):
        start, end = starts[position], starts[position + 1]
        header = lines[start].split("\t")
        if "site_no" not in header or "datetime" not in header:
            continue
        site_col = header.index("site_no")
        date_col = header.index("datetime")
        candidates = [
            i for i, name in enumerate(header)
            if "00060" in name and "00003" in name and not name.endswith("_cd")
        ]
        if not candidates:
            candidates = [
                i for i, name in enumerate(header)
                if "00060" in name and not name.endswith("_cd")
            ]
        if not candidates:
            continue
        value_col = candidates[0]
        parsed = [line.split("\t") for line in lines[start + 2 : end]]
        parsed = [row for row in parsed if len(row) > max(site_col, date_col, value_col)]
        if not parsed:
            continue
        gage = str(parsed[0][site_col]).zfill(8)
        dates = pd.to_datetime([row[date_col] for row in parsed], errors="coerce")
        values = pd.to_numeric(pd.Series([row[value_col] for row in parsed]), errors="coerce").to_numpy()
        frame = pd.DataFrame({"date": dates, "Q_cfs": values})
        frame = frame[
            frame.date.between(START, END) & np.isfinite(frame.Q_cfs)
        ].copy()
        if len(frame):
            yield gage, frame


def build_discharge_qc() -> tuple[pd.DataFrame, pd.DataFrame]:
    cached_qc = OUT / "daily_discharge_qc.csv"
    cached_monthly = OUT / "monthly_discharge.parquet"
    if cached_qc.exists() and cached_monthly.exists():
        return pd.read_csv(cached_qc, dtype={"GAGE_ID": str}), pd.read_parquet(cached_monthly)

    valid_inventory = inventory()
    wanted = set(valid_inventory.GAGE_ID)
    first: dict[str, pd.Timestamp] = {}
    last: dict[str, pd.Timestamp] = {}
    valid_days: defaultdict[str, int] = defaultdict(int)
    monthly_frames: list[pd.DataFrame] = []

    paths = sorted((ROOT / "usgs_dv_00060_1980_01_01_to_2025_12_31" / "raw_rdb_gz").glob("*.rdb.gz"))
    for file_number, path in enumerate(paths, start=1):
        for gage, frame in read_rdb_sections(path):
            if gage not in wanted:
                continue
            first_date = frame.date.min()
            last_date = frame.date.max()
            first[gage] = min(first.get(gage, first_date), first_date)
            last[gage] = max(last.get(gage, last_date), last_date)
            valid_days[gage] += len(frame)
            frame["date"] = frame.date.dt.to_period("M").dt.to_timestamp()
            monthly = frame.groupby("date").Q_cfs.agg(["mean", "count"]).reset_index()
            monthly.insert(0, "GAGE_ID", gage)
            monthly.rename(columns={"mean": "Q_cfs", "count": "n_days_Q"}, inplace=True)
            monthly.loc[monthly.n_days_Q < 20, "Q_cfs"] = np.nan
            monthly_frames.append(monthly)
        if file_number % 20 == 0:
            print(f"  discharge files {file_number}/{len(paths)}", flush=True)

    qc_rows = []
    for gage in sorted(wanted):
        if gage not in first:
            qc_rows.append({"GAGE_ID": gage, "valid_daily_values": 0, "eligible_daily_qc": False})
            continue
        span_days = (last[gage] - first[gage]).days + 1
        record_years = span_days / 365.2425
        completeness = valid_days[gage] / span_days
        qc_rows.append({
            "GAGE_ID": gage,
            "first_valid_date": first[gage],
            "last_valid_date": last[gage],
            "record_span_days": span_days,
            "record_years": record_years,
            "valid_daily_values": valid_days[gage],
            "daily_completeness_within_span": completeness,
            "eligible_daily_qc": record_years >= MIN_RECORD_YEARS and completeness >= MIN_DAILY_COMPLETENESS,
        })
    qc = pd.DataFrame(qc_rows).merge(valid_inventory, on="GAGE_ID", how="left", validate="one_to_one")
    qc.to_csv(cached_qc, index=False)

    eligible = set(qc.loc[qc.eligible_daily_qc, "GAGE_ID"])
    monthly = pd.concat(monthly_frames, ignore_index=True)
    monthly = monthly[monthly.GAGE_ID.isin(eligible)].sort_values(["GAGE_ID", "date"])
    monthly.to_parquet(cached_monthly, index=False, compression="zstd")
    return qc, monthly


def aggregate_climate(candidate_gages: set[str]) -> pd.DataFrame:
    cached = OUT / "monthly_climate.parquet"
    if cached.exists():
        return pd.read_parquet(cached)
    pieces = []
    for filename, value_column, output_column in (
        ("basin_precipitation.csv", "precipitation_mm", "P_mm"),
        ("basin_pet.csv", "pet_mm", "PET_mm"),
    ):
        monthly_parts = []
        for number, chunk in enumerate(pd.read_csv(
            ROOT / filename,
            usecols=["GAGE_ID", "date", value_column],
            dtype={"GAGE_ID": str},
            chunksize=3_000_000,
        ), start=1):
            chunk["GAGE_ID"] = chunk.GAGE_ID.str.zfill(8)
            chunk = chunk[chunk.GAGE_ID.isin(candidate_gages)].copy()
            if len(chunk):
                chunk["date"] = pd.to_datetime(chunk.date).dt.to_period("M").dt.to_timestamp()
                monthly_parts.append(
                    chunk.groupby(["GAGE_ID", "date"])[value_column]
                    .agg(["sum", "count"]).reset_index()
                )
            if number % 5 == 0:
                print(f"  {output_column} chunks {number}", flush=True)
        frame = pd.concat(monthly_parts, ignore_index=True)
        frame = frame.groupby(["GAGE_ID", "date"], as_index=False).agg({"sum": "sum", "count": "sum"})
        frame.rename(columns={"sum": output_column, "count": f"n_days_{output_column}"}, inplace=True)
        frame.loc[frame[f"n_days_{output_column}"] < 25, output_column] = np.nan
        pieces.append(frame)
    climate = pieces[0].merge(pieces[1], on=["GAGE_ID", "date"], how="outer", validate="one_to_one")
    climate.to_parquet(cached, index=False, compression="zstd")
    return climate


def lmom(values: np.ndarray) -> tuple[float, float, float]:
    values = np.sort(values[np.isfinite(values)])
    n = len(values)
    if n < 3:
        return np.nan, np.nan, np.nan
    j = np.arange(n, dtype=float)
    b0 = values.mean()
    b1 = np.sum(values * j) / (n * (n - 1))
    b2 = np.sum(values * j * (j - 1)) / (n * (n - 1) * (n - 2))
    return b0, 2 * b1 - b0, 6 * b2 - 6 * b1 + b0


def glo_fit(values: np.ndarray) -> tuple[float, float, float] | None:
    """Hosking generalized-logistic parameters from sample L-moments."""
    l1, l2, l3 = lmom(values)
    if not np.isfinite(l1) or not np.isfinite(l2) or l2 <= 0:
        return None
    k = -float(np.clip(l3 / l2, -0.99, 0.99))
    if abs(k) < 1e-7:
        return l1, l2, 0.0
    alpha = l2 * np.sin(np.pi * k) / (np.pi * k)
    xi = l1 - alpha * (1.0 / k - np.pi / np.sin(np.pi * k))
    if not np.isfinite(alpha) or alpha <= 0:
        return None
    return xi, alpha, k


def glo_cdf(values: np.ndarray, parameters: tuple[float, float, float]) -> np.ndarray:
    xi, alpha, k = parameters
    y = (np.asarray(values, dtype=float) - xi) / alpha
    if abs(k) < 1e-7:
        result = 1.0 / (1.0 + np.exp(-y))
    else:
        base = 1.0 - k * y
        result = np.full(base.shape, 1.0 if k > 0 else 0.0, dtype=float)
        valid = base > 0
        result[valid] = 1.0 / (1.0 + np.power(base[valid], 1.0 / k))
    return np.clip(result, 1e-8, 1 - 1e-8)


def gamma_fit(values: np.ndarray) -> tuple[float, float] | None:
    positive = values[np.isfinite(values) & (values > 0)]
    if len(positive) < 5:
        return None
    l1, l2, _ = lmom(positive)
    if l1 <= 0 or l2 <= 0:
        return None
    t = l2 / l1
    if t < 0.5:
        z = np.pi * t * t
        shape = (1 - 0.3080 * z) / (z - 0.05812 * z * z + 0.01765 * z**3)
    else:
        z = 1 - t
        shape = (0.7213 * z - 0.5947 * z * z) / (1 - 2.1817 * z + 1.2113 * z * z)
    if not np.isfinite(shape) or shape <= 0:
        return None
    return shape, l1 / shape


def standardized_index(
    values: np.ndarray,
    months: np.ndarray,
    calibration: np.ndarray,
    distribution: str,
) -> np.ndarray:
    output = np.full(len(values), np.nan, dtype=float)
    for month in range(1, 13):
        selected = months == month
        calibration_values = values[selected & calibration & np.isfinite(values)]
        if len(calibration_values) < 10:
            continue
        target = values[selected]
        if distribution == "glo":
            parameters = glo_fit(calibration_values)
            if parameters is None:
                continue
            probability = glo_cdf(target, parameters)
        elif distribution == "gamma":
            zero_probability = float((calibration_values <= 0).mean())
            parameters = gamma_fit(calibration_values)
            if parameters is None:
                continue
            shape, scale = parameters
            probability = np.where(
                target > 0,
                zero_probability + (1 - zero_probability) * special.gammainc(shape, np.clip(target, 0, None) / scale),
                zero_probability / 2,
            )
        elif distribution == "empirical":
            ordered = np.sort(calibration_values)
            rank = np.searchsorted(ordered, target, side="right")
            probability = (rank + 0.5) / (len(ordered) + 1.0)
        else:
            raise ValueError(distribution)
        probability = np.where(np.isfinite(target), np.clip(probability, 1e-8, 1 - 1e-8), np.nan)
        output[selected] = stats.norm.ppf(probability)
    return np.clip(output, -4, 4)


def calendar_support(values: np.ndarray, months: np.ndarray, calibration: np.ndarray) -> int:
    return int(min(np.sum(calibration & (months == month) & np.isfinite(values)) for month in range(1, 13)))


def validate_glo() -> dict[str, float]:
    sample = np.array([-4.2, -2.1, -1.3, -0.7, 0.0, 0.4, 1.2, 2.8, 5.4])
    _, l2, _ = lmom(sample)
    parameters = glo_fit(sample)
    if parameters is None:
        raise AssertionError("GLO validation fit failed")
    _, alpha, k = parameters
    reconstructed_l2 = alpha if abs(k) < 1e-7 else alpha * k * np.pi / np.sin(k * np.pi)
    if not np.isclose(reconstructed_l2, l2, rtol=1e-12, atol=1e-12):
        raise AssertionError("GLO scale does not reproduce the sample second L-moment")
    return {"sample_l2": float(l2), "reconstructed_l2": float(reconstructed_l2), "shape_k": float(k)}


def build_indices(qc: pd.DataFrame, discharge: pd.DataFrame, climate: pd.DataFrame) -> None:
    daily_qc = qc[qc.eligible_daily_qc].copy()
    candidate = set(daily_qc.GAGE_ID)
    merged = discharge.merge(climate, on=["GAGE_ID", "date"], how="outer", validate="one_to_one")
    merged = merged[merged.GAGE_ID.isin(candidate)].copy()
    rows = []
    support_rows = []
    full_index = pd.date_range("1980-01-01", "2025-12-01", freq="MS")
    for number, (gage, group) in enumerate(merged.groupby("GAGE_ID", sort=False), start=1):
        group = group.set_index("date").reindex(full_index)
        precipitation = group.P_mm.astype(float)
        pet = group.PET_mm.astype(float)
        discharge_values = group.Q_cfs.astype(float)
        months = full_index.month.to_numpy()
        fixed = np.asarray(full_index <= CAL_END)
        all_period = np.ones(len(full_index), dtype=bool)
        record: dict[str, object] = {"GAGE_ID": gage, "date": full_index}
        supports: dict[str, object] = {"GAGE_ID": gage}

        water_balance = precipitation - pet
        for scale in (3, 6):
            wb = water_balance.rolling(scale, min_periods=scale).sum().to_numpy()
            pp = precipitation.rolling(scale, min_periods=scale).sum().to_numpy()
            record[f"SPEI_{scale}_fixed"] = standardized_index(wb, months, fixed, "glo")
            record[f"SPEI_{scale}"] = standardized_index(wb, months, all_period, "glo")
            record[f"SPI_{scale}_fixed"] = standardized_index(pp, months, fixed, "gamma")
            record[f"SPI_{scale}"] = standardized_index(pp, months, all_period, "gamma")
            supports[f"SPEI_{scale}_min_calendar_support"] = calendar_support(wb, months, fixed)

        for scale in (1, 3, 6):
            qq = discharge_values.rolling(scale, min_periods=scale).sum().to_numpy()
            record[f"SSI_{scale}_fixed"] = standardized_index(qq, months, fixed, "gamma")
            record[f"SSI_{scale}"] = standardized_index(qq, months, all_period, "gamma")
            record[f"SSI_{scale}_empirical_fixed"] = standardized_index(qq, months, fixed, "empirical")
            supports[f"SSI_{scale}_min_calendar_support"] = calendar_support(qq, months, fixed)

        rows.append(pd.DataFrame(record))
        support_rows.append(supports)
        if number % 500 == 0:
            print(f"  indices {number}/{len(candidate)}", flush=True)

    indices = pd.concat(rows, ignore_index=True)
    support = pd.DataFrame(support_rows)
    required = [
        "SPEI_3_min_calendar_support", "SPEI_6_min_calendar_support",
        "SSI_1_min_calendar_support", "SSI_3_min_calendar_support", "SSI_6_min_calendar_support",
    ]
    support["eligible_calibration25"] = support[required].min(axis=1) >= MIN_CALENDAR_MONTH_SUPPORT
    population = daily_qc.merge(support, on="GAGE_ID", how="left", validate="one_to_one")
    population.to_csv(OUT / "audited_study_population.csv", index=False)

    keep = set(population.loc[population.eligible_calibration25, "GAGE_ID"])
    indices["primary_population"] = indices.GAGE_ID.isin(keep)
    for column in indices.columns:
        if column not in {"GAGE_ID", "date", "primary_population"}:
            indices[column] = pd.to_numeric(indices[column], errors="coerce").astype("float32")
    indices.to_parquet(OUT / "audited_indices.parquet", index=False, compression="zstd")
    merged.to_parquet(OUT / "audited_monthly_inputs.parquet", index=False, compression="zstd")

    validation = validate_glo()
    validation.update({
        "n_conterminous_inventory_gauges": int(len(qc)),
        "n_daily_qc_gauges": int(daily_qc.GAGE_ID.nunique()),
        "n_primary_calibration25_gauges": int(len(keep)),
        "primary_population_rule": "daily span >=30 years; >=90% valid daily Q within span; >=25 valid accumulated calibration values in every calendar month for SPEI-3/6 and SSI-1/3/6",
    })
    (OUT / "input_index_validation.json").write_text(json.dumps(validation, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["all", "qc", "climate", "indices"], default="all")
    args = parser.parse_args()
    qc, discharge = build_discharge_qc()
    print(f"Daily-QC eligible gauges: {int(qc.eligible_daily_qc.sum()):,}", flush=True)
    if args.stage == "qc":
        return
    climate = aggregate_climate(set(qc.loc[qc.eligible_daily_qc, "GAGE_ID"]))
    if args.stage == "climate":
        return
    build_indices(qc, discharge, climate)
    print(f"Audited inputs and indices written to {OUT}", flush=True)


if __name__ == "__main__":
    main()
