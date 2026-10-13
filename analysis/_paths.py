"""Shared path configuration for the numbered analysis scripts."""

from __future__ import annotations

import os
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = Path(os.environ.get("RDB_PROJECT_ROOT", REPOSITORY_ROOT / "source_data")).expanduser().resolve()
DATA_ROOT = Path(
    os.environ.get(
        "RDB_DATA_ROOT",
        REPOSITORY_ROOT / "data" / "river-drought-boundaries-data-v1.0.0",
    )
).expanduser().resolve()
OUTPUTS = Path(
    os.environ.get("RDB_OUTPUTS", DATA_ROOT / "data" / "analysis_outputs")
).expanduser().resolve()
WORK = Path(os.environ.get("RDB_WORK_DIR", REPOSITORY_ROOT / "work")).expanduser().resolve()
BASIN_ATTRIBUTES = DATA_ROOT / "data" / "basins" / "basin_attributes_used.csv"
NLDAS_MONTHLY = DATA_ROOT / "data" / "nldas" / "nldas_root_zone_monthly.parquet"
