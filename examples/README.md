# Example Inputs

This directory is reserved for safe demonstration inputs.

Do not place an actual participant recording here unless its redistribution is explicitly authorized. Preferred options are:

- Synthetic recordings that follow the required schema.
- Header-only CSV templates.
- A short, fully deidentified example specifically approved for release.

The demonstration requires:

```text
imu_template.csv
biomarkers_template.csv
```

The header-only templates in this directory document the required fields but cannot generate a prediction. A runnable example must contain at least 60 seconds of overlapping timestamps and valid measurements.

See `docs/data_format.md` for units, timestamp requirements, participant characteristics, and intermediate NPZ formats.
