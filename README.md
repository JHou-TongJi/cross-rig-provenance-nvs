# Cross-rig provenance for camera and LiDAR reprojection

A per-pixel provenance contract and an evidence profile for reprojecting a
recorded multi-camera corpus into a sensor rig that was never driven, with the
inference pipeline and the derived evidence behind them.

## Overview

A recorded multi-camera corpus stops matching its vehicle as soon as the rig is
re-specified — a camera moves, gets a different lens, or the mounting changes.
Reprojecting the corpus into the new rig is the obvious response. The question
this work addresses is how anyone would know whether the result may be trusted,
when the target platform was never driven and so no target-view reference image
exists to score against.

## What the method does

A **provenance contract** assigns every delivered pixel to exactly one of three
classes — `observed` where a calibrated correspondence supports it,
`gate-recovered` where geometry can be shown to support it, `generated`
otherwise — and enforces that partition in the compositing code rather than
documenting it afterwards.

Observed pixels are locked byte-exactly on the rectified internal raster the
lock is defined over. Export redistorts that raster by interpolation, so byte
equality is claimed before the resampling and not after it. No visual-only
module may write depth, pose or any reported geometric quantity.

Three label transitions are specified, because a partition without transition
rules is underspecified the moment a stage writes twice: observed admits no
transition; a gate pixel a visual-only module writes becomes generated and
loses its depth eligibility; generated is terminal. The rule is conservative by
construction, so the observed class can only shrink under processing.

An **evidence profile** replaces the photometric score that is undefined in
this regime. It reports observed coverage, gate acceptance under an explicit
cycle test, generated fraction, locked-pixel compliance, per-camera risk and
temporal spread, each with a defined support and a stated measurement status.

## Results

Numbers are in `evidence/`; every one is derived by script from the raw
evaluation data rather than transcribed.

- The geometry branch reconstructs all seven target cameras under two
  parameterized rigs at 94.07% and 96.66% mean coverage.
- The learned branch reaches 98.24% mean observed coverage over 21 shifted-pose
  cases while modifying zero locked pixels.
- Gate acceptance is reported both ways, because the two differ: 44.90% as an
  unweighted per-camera mean and 37.77% pooled over all hole pixels. Gate fill
  is denominated by the hole, and hole counts vary by case.
- A seven-camera mean of 4.94% vehicle-region risk conceals 24.23% on one
  camera, a factor of 4.9.

## Honest limits

These are consistency and traceability audits, not measurements of fidelity.
They do not establish reconstruction accuracy, geometric correctness, or
fitness for training a perception stack.

- The learned results are same-name small-baseline reprojections, in which the
  source camera nearest the target stays in the input. They do not demonstrate
  synthesis of an unseen viewpoint.
- Both target rigs are parameterized configuration files, not measured vehicle
  calibrations, so no result characterizes a real platform.
- One sequence, one learned-branch key frame, one daytime condition.
- No baseline, no component ablation, no held-out depth, no downstream test.

## Layout

```
code/       inference pipeline (GCR-NVS), rig configurations, environment files
evidence/   derived evaluation results: CSV, redacted JSON run reports
docs/       sensor, rig and data-scope documentation (see docs/README.md)
tools/      contract property checker
```

## Install and deploy

See **[DEPLOY.md](DEPLOY.md)**. In short: the contract property checker runs on
a clean clone with no dependencies and no data; the pipeline needs a GPU plus
the road sequence and frozen checkpoints, which are not redistributable.

## What is not here, and why

The reference sequence is used under a challenge data-use agreement:

| Withheld | Reason |
|---|---|
| Raw RGB imagery and LiDAR point clouds | Data-use agreement; not redistributable |
| `camera_config.json`, `camera_intric.yaml` | Manufacturer calibration of the source vehicle |
| `target_camera_module_7v.json` | Re-packages the same supplied extrinsics |
| Model weights (`*.pt`) | Referenced by controlled path; ~608 MB |
| Rig pose blocks inside JSON run reports | Derived from the withheld calibration; redacted in place |
| The manuscript | Under review; this repository is the code and the evidence |

Redaction is by field, not by dropping files: every coverage figure in
`evidence/` survives, and the pose matrices those figures were computed with
read `"withheld: derived from source calibration not redistributed"`.

Two further gaps, stated rather than left to be discovered:

- **The L4-SE geometry-branch source is not in this snapshot.** The tree
  carries an empty placeholder; its documentation is in `code/docs/` and the
  results it produced are in `evidence/`.
- **Camera focal lengths are published**, as they are in the article.

## Licensing

Apache-2.0, see [LICENSE](LICENSE). It covers the code and documentation
written for this project. It does not cover the third-party models the
pipeline loads, each of which carries its own upstream licence — see
[third_party.md](third_party.md) — nor the reference sequence, which is not
redistributed at all.

## Status

The associated article is under review. Citation details will be added on
acceptance.
