# Sample-data dictionary

All records refer to USGS GAGE_ID `01011000`. Identifiers are strings and must not be converted to integers because leading zeros are meaningful.

## `monthly_indices_sample.csv`

| Field | Meaning |
|---|---|
| `GAGE_ID` | Public USGS streamgage identifier |
| `date` | First day of the monthly time step |
| `SPEI_3_fixed`, `SPEI_6_fixed` | Standardized Precipitation Evapotranspiration Index accumulated over 3 or 6 months and fitted over 1980–2010 |
| `SSI_3_fixed`, `SSI_6_fixed` | Standardized Streamflow Index accumulated over 3 or 6 months and fitted over 1980–2010 |

## `basin_attributes_sample.csv`

| Field | Meaning |
|---|---|
| `HUC02` | Two-digit water-resource region |
| `AGGECOREGION` | Aggregate GAGES-II ecoregion |
| `reference_class` | Minimally disturbed or potentially human-influenced basin class |
| `DRAIN_SQKM` | Drainage area (km²) |
| `LAT_GAGE`, `LNG_GAGE` | Streamgage coordinates (decimal degrees) |
| `aridity` | Basin aridity index used in stratified analysis |
| `WTDEPAVE` | Mean water-table depth (m) |
| `SNOW_PCT_PRECIP` | Snowfall as a percentage of total precipitation |

## `expected_events_sample.csv`

Each row is an eligible meteorological event. The table records its accumulation scale and dates, direct river state at meteorological onset, any assigned new river onset, the resulting onset pathway, and direct river state at meteorological termination.

The sample is distributed under CC BY 4.0. Upstream USGS, gridMET and GAGES-II terms continue to apply.
