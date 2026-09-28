# Evidence

Every number reported in the associated article is derived from the files here
by script. Nothing in this directory is transcribed by hand.

## What is measured and what is a proxy

Read `联合方案指标口径与自测计划.md` first. It records which quantities are
measurements and which are proxies — the distinction the provenance contract
exists to keep visible. In short: coverage, gate acceptance, generated fraction
and locked-pixel compliance are consistency and traceability audits. None is a
fidelity measurement against an independent reference, and no combination of
them establishes that a delivered colour or depth is correct.

## Per-case and per-camera results

| File | Contents |
|---|---|
| `gcr-nvs/quality_metrics.csv` | The 21 shifted-pose cases: observed coverage, gate fill of hole, generated fraction, vehicle-risk fraction, locked pixels modified |
| `quality_metrics.csv` | 27 rows across both branches and metric scopes |
| `current_rig_quality_metrics.csv` | Seven-camera geometry audit at frame 40 |
| `camera_support.csv` | Per-camera LiDAR support: HFOV, sensor megapixels, returns in frame, returns per megapixel, unknown coverage |
| `self_evaluation_summary.csv` | Summary metrics with scope and interpretation |
| `data_lineage.csv` | The eleven lineage classes and what each may export |

A caution on `camera_support.csv`: `returns_per_mp` is counted on each camera's
**native** raster, while the coverage it is compared against comes from an audit
run at 1920 × 1080 for every camera. Relating the two requires recounting
returns per megapixel on the audit raster. Mixing the two rasters changes both
the magnitude and the apparent conclusion, so the article reports the recounted
figures.

## Result tables

`tables/` holds every table from the article as CSV — header row plus data,
captions left behind. `table_01` to `table_14` follow the article's numbering.

Three further files carry row-level values the article only summarises:

| File | Contents |
|---|---|
| `tables/per_case_all_21_shifted_pose.csv` | Complete per-case results for all 21 cases |
| `tables/per_camera_lidar_support.csv` | Row-level support values behind the two correlations |
| `tables/temporal_20_frame_coverage.csv` | The twenty frame-level coverage values, long form |

`temporal_20_frame_coverage.csv` reproduces the reported mean of 94.42%
exactly; its standard deviation reads 0.44 as a population figure and 0.45 as a
sample figure, which is the only difference.

## Run reports

`raw_reports/` holds the JSON reports the evaluations produced. **The rig pose
blocks inside them are redacted field by field**, because they derive from the
source vehicle's manufacturer calibration, which is not redistributable. The
redaction replaces each matrix with
`"withheld: derived from source calibration not redistributed"` and leaves
every coverage figure intact, so the reports still support the numbers they
were cited for.

## Failure cases

`failure_cases.md` and `gcr-nvs/failure_cases.md` record where the method
breaks and the boundary of applicability. They are part of the evidence, not an
appendix to it.

## What is not here

Raw RGB imagery, LiDAR point clouds, the source vehicle's manufacturer
calibration and the model weights. See the repository README for the full list
and the reasons.
