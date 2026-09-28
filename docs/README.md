# Documentation index

Most documents in this repository were written in Chinese during the project
and are published as they were rather than retranslated, so that nothing is
lost or quietly reworded in the move to English. This index says what each one
covers.

## `docs/` — sensors, rigs and data scope

| File | Covers |
|---|---|
| `数据来源说明.md` | Where the reference data came from and the terms it is used under |
| `源车型传感器说明.md` | The source vehicle's sensor suite: cameras, LiDAR, mounting |
| `目标小车传感器说明.md` | The target L4 logistics vehicle's sensor suite |
| `开放式目标车型与7V相机模组参数说明.md` | The open target-vehicle preset and the seven-camera module parameters |
| `联合方案数据与目标传感器说明.md` | Combined data and target-sensor description across both branches |

## `code/docs/` — reproduction and environment

| File | Covers |
|---|---|
| `GCR-NVS复现说明.md` | Reproduction notes for the inference-only GCR-NVS branch |
| `l4-se复现说明.md` | Sampling-based reproduction notes for the L4-SE branch |
| `代码与环境说明.md` | L4-SE code layout, environment and reproduction |
| `model_manifest.md` | External models, versions and delivery method. Note: the licence column is a recorded requirement, not yet filled in — see `third_party.md` |
| `大文件手工复制清单.md` | Large files kept on the data disk and copied by hand, never committed |

## `evidence/` — measured results and their limits

| File | Covers |
|---|---|
| `实测结果报告.pdf` | Measured-results report |
| `gcr-nvs/实测结果说明.md` | What each measured result in `gcr-nvs/` is and how it was produced |
| `failure_cases.md` | Failure cases and the boundary of applicability |
| `gcr-nvs/failure_cases.md` | Failure cases and known limits for the GCR-NVS branch |
| `联合方案指标口径与自测计划.md` | Metric definitions and the self-evaluation plan, including which quantities are proxies |

## A note on the metric documents

`联合方案指标口径与自测计划.md` is worth reading before any number in
`evidence/`. It records which quantities are measurements and which are
proxies — the distinction the provenance contract exists to keep visible, and
the reason the associated article reports an evidence profile rather than a
photometric score.
