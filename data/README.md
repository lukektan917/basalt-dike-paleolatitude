# Data

Ten magnetometer transects walked across a basalt dike in the Middlesex Fells,
Massachusetts, on 13 September 2025. Each CSV is one traverse, recorded with a
phone magnetometer at 100 Hz:

| column | meaning |
|---|---|
| `time` | seconds since the recording started |
| `Bx`, `By`, `Bz` | field components in microtesla; `Bz` is the one used here |

Files are named `magnetometer_<date>_<time>.csv`; sorting by filename puts them
in the order they were walked.

Place the survey CSVs in `data/high_50cm/` to reproduce the published figures:

```
python -m dike.run --data data/high_50cm --out figures
```

If you do not have the field data, every figure can still be regenerated from
simulated transects with a known answer:

```
python -m dike.run --synthetic --out figures
```
