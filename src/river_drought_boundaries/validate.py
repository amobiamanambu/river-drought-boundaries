"""Validate the versioned research dataset and expected scientific invariants."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


EXPECTED = {
    3: {
        "events": 14755,
        "preexisting_percent": 9.698407319552697,
        "active_percent": 44.38495425279566,
        "omitted_percent": 53.51996397478235,
    },
    6: {
        "events": 12372,
        "preexisting_percent": 8.874878758486906,
        "active_percent": 49.52311671516328,
        "omitted_percent": 45.78676601028915,
    },
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_checksums(root: Path) -> list[str]:
    failures: list[str] = []
    checksum_file = root / "SHA256SUMS"
    if not checksum_file.exists():
        return ["SHA256SUMS is missing"]
    for line in checksum_file.read_text(encoding="utf-8").splitlines():
        expected, relative = line.split("  ", 1)
        path = root / relative
        if not path.exists():
            failures.append(f"missing file: {relative}")
        elif sha256(path) != expected:
            failures.append(f"checksum mismatch: {relative}")
    return failures


def validate_dataset(root: Path) -> dict[str, object]:
    outputs = root / "data" / "analysis_outputs"
    primary = pd.read_parquet(outputs / "revised_state_catalogue_primary.parquet")
    primary["GAGE_ID"] = primary.GAGE_ID.astype(str).str.zfill(8)
    attributes = pd.read_csv(root / "data" / "basins" / "basin_attributes_used.csv", dtype={"GAGE_ID": str, "HUC02": str})
    attributes["GAGE_ID"] = attributes.GAGE_ID.str.zfill(8)

    failures = verify_checksums(root)
    checks: list[dict[str, object]] = []

    def check(name: str, passed: bool, observed: object, expected: object) -> None:
        checks.append({"name": name, "passed": bool(passed), "observed": observed, "expected": expected})
        if not passed:
            failures.append(name)

    check("primary rows", len(primary) == 27127, len(primary), 27127)
    duplicate_count = int(primary.duplicated(["GAGE_ID", "timescale", "met_onset"]).sum())
    check("primary event keys are unique", duplicate_count == 0, duplicate_count, 0)
    check("primary basin attributes", len(attributes) == 3571, len(attributes), 3571)
    uncovered = len(set(primary.GAGE_ID) - set(attributes.GAGE_ID))
    check("event basins have attributes", uncovered == 0, uncovered, 0)

    for scale, expected in EXPECTED.items():
        group = primary[primary.timescale.eq(scale)]
        at_risk = group[~group.river_preexisting_at_met_onset]
        values = {
            "events": len(group),
            "preexisting_percent": 100 * group.river_preexisting_at_met_onset.mean(),
            "active_percent": 100 * group.river_active_at_met_termination.mean(),
            "omitted_percent": 100 * at_risk.assigned_hyd_onset.isna().mean(),
        }
        for metric, target in expected.items():
            observed = values[metric]
            passed = observed == target if metric == "events" else np.isclose(observed, target, atol=1e-10)
            check(f"{scale}-month {metric}", passed, observed, target)

    return {"all_checks_passed": not failures, "failures": failures, "checks": checks}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_root", type=Path, help="Extracted Zenodo dataset directory")
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON")
    arguments = parser.parse_args()
    report = validate_dataset(arguments.data_root.resolve())
    if arguments.json:
        print(json.dumps(report, indent=2))
    else:
        status = "PASS" if report["all_checks_passed"] else "FAIL"
        print(f"Dataset validation: {status}")
        print(f"Scientific checks: {sum(item['passed'] for item in report['checks'])}/{len(report['checks'])}")
        if report["failures"]:
            for failure in report["failures"]:
                print(f"- {failure}")
    raise SystemExit(0 if report["all_checks_passed"] else 1)


if __name__ == "__main__":
    main()
