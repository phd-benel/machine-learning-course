# README_DATAPROFILIING — Data Profiling (filtered_elysee.csv)

## Scope

- **Input**: `data/processed/filtered_elysee.csv`
- **Perimeter**: Élysée
- **Goal**: document data health *before* running `0.5_transform.py`

## 1) Dataset overview

- **Rows**: `<N>`
- **Columns**: `<K>`
- **Identifier column**: `<id column name>`
- **Profiling date**: `<YYYY-MM-DD>`

## 2) Missing values (NaN audit)

| Column | % missing | Decision (drop / impute / keep) | Rationale |
|---|---:|---|---|
| price | `<...>` | `<...>` | `<...>` |
| reviews_per_month | `<...>` | `<...>` | `<...>` |

## 3) Type & format issues

- **price**: raw examples (e.g. `"$120.00"`), conversion rule, parsing errors found.
- Other columns that need conversion: `<...>`

## 4) Numeric ranges (sanity check)

| Feature | min | max | median | Notes |
|---|---:|---:|---:|---|
| price | `<...>` | `<...>` | `<...>` | `<outliers?>` |
| availability_365 | `<...>` | `<...>` | `<...>` | `<...>` |

## 5) Outliers / anomalies

- Example IDs / rows: `<...>`
- Proposed rules: capping thresholds / removals, and why.

## 6) Decisions that drive Transform (rules)

- **Drop rules**: `<...>`
- **Imputation rules**: `<...>`
- **Normalization rules**: `<...>`
- **Logging**: what errors / anomalies are logged during transform.

