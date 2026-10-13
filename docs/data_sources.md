# Source data

The complete upstream rebuild uses four public data systems. The companion data release redistributes only derived basin-month values and the specific static attributes used in the analyses.

## USGS National Water Information System

- Role: daily mean streamflow.
- Study extraction: 1 January 1980 through 31 December 2025.
- Parameter: discharge, USGS parameter code 00060.
- Access: https://waterdata.usgs.gov/nwis
- Primary identifier: `GAGE_ID`, stored as text to preserve leading zeros; valid identifiers in this study contain eight to ten digits.

The input-reconstruction script accepts gzip-compressed USGS RDB files organized under `raw_rdb_gz/`. It identifies the daily-value column by parameter and statistic codes, converts valid observations to monthly means and records the number of daily observations in each month.

## USGS GAGES-II

- Role: conterminous basin inventory, drainage area, region, aggregate ecoregion, reference status, hydroclimatic attributes, water-table depth and map boundaries.
- Access and citation: https://doi.org/10.3133/ds609

The release contains `basin_attributes_used.csv`, which consolidates only the attributes used by the analysis. The original GAGES-II tables remain the authoritative source.

## gridMET

- Role: daily precipitation and ASCE Penman–Monteith reference evapotranspiration at approximately 4-km resolution.
- Access: https://www.climatologylab.org/gridmet.html
- Citation: Abatzoglou, J. T. Development of gridded surface meteorological data for ecological applications and modelling. *International Journal of Climatology* 33, 121–131 (2013).

Daily basin means are summed to monthly totals only when at least 25 daily values are present.

## NLDAS-2 Noah

- Role: supplementary root-zone soil-moisture analysis.
- Product: NLDAS_NOAH0125_M.002.
- Access: https://disc.gsfc.nasa.gov/datasets/NLDAS_NOAH0125_M_002/summary
- Citation: Xia, Y. et al. Continental-scale water and energy flux analysis and validation for the North American Land Data Assimilation System project phase 2. *Journal of Geophysical Research: Atmospheres* 117, D03109 (2012).

The released NLDAS table contains area-weighted basin-month values rather than the original gridded files.

## Local layout for a complete upstream rebuild

Set `RDB_PROJECT_ROOT` to a directory containing:

```text
basin_inventory.csv
gagesii_csv/
gridmet_basin_daily/
usgs_dv_00060_1980_01_01_to_2025_12_31/raw_rdb_gz/
Data/External/Basin_Subsets/nldas_storage_monthly_3606_basins_1980_2025.parquet
```

The versioned Zenodo release is the preferred input for exact analytical reproduction. A rebuild from upstream sources may differ if providers revise observations after the archived extraction date.
