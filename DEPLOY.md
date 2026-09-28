# Deploy and reproduce

Two things in this repository run, and they have very different requirements.

| What you want to run | GPU | Road data | Model weights |
|---|---|---|---|
| Contract property checker | no | no | no |
| The GCR-NVS pipeline | yes | yes | yes |

The first runs on a clean clone in seconds. The second cannot be run from this
repository alone, for reasons set out at the end.

---

## 1. Contract property checker

The only part that needs nothing but Python. It checks the provenance
contract's six invariants against a reference implementation of the
specification, with negative controls that break each rule in turn and require
the matching invariant to fail — a property that cannot fail is not a test.

```bash
python tools/contract_properties.py
```

Expected final line:

```
PASS: 6 properties hold over 20000 frames; all 4 injected violations were detected
```

Python 3.9 or newer, no third-party packages. Options: `--cases N`, `--seed S`.

The six invariants: the partition is exhaustive and mutually exclusive; an
observed pixel never receives a non-zero modification budget; a generated pixel
never exports depth; no transition gains evidence; a visual-only write demotes
a gate pixel and removes its depth eligibility; a resampled pixel takes its
lowest-evidence contributor.

What this establishes is narrow and worth stating precisely. The rules are
mutually consistent and each invariant is falsifiable. It says nothing about
whether the pipeline implements them — the reference implementation shares no
code with GCR-NVS, which is what makes agreement evidence rather than
tautology, and equally what leaves the deployed behaviour unmeasured.

---

## 2. The GCR-NVS pipeline

`code/gcr-nvs-inference-only-20260826/` holds the inference implementation,
configurations and environment files. The released configuration is
`configs/releases/t0_uniview_a_final.yaml`.

```bash
python -m venv .venv && . .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r code/requirements.gcr-core.txt      # torch 2.4.0, CUDA 12.4
pip install -r code/requirements.gcr-optional.txt  # optional visual components
```

Tested under Python 3.11 with CUDA 12.4 on Linux; the CUDA runtime arrives with
the PyTorch wheels. Per-component requirement files are also provided
(`requirements.l4-se.txt`, `requirements-da3.txt`, `requirements-3dgs.txt`,
`requirements-viewer.txt`) so a partial environment can be built.

`third_party.md` lists every external model the pipeline loads.

### What you cannot run from this repository

Three things are needed and are not here:

- **The road-collection sequence.** Used under a challenge data-use agreement
  that forbids redistribution. Requests go to the data provider.
- **The source vehicle's manufacturer calibration.** Withheld for the same
  reason. The rig pose blocks inside the JSON run reports are redacted field by
  field, so every coverage figure in `evidence/` survives while the matrices
  behind them do not.
- **The frozen checkpoints.** Referenced by controlled path, about 608 MB,
  available from the corresponding author on reasonable request and subject to
  the upstream licences in `third_party.md`.

### What reproduces and what does not

The **geometry branch** carries no learned component, so its results reproduce
in principle given the sequence, the calibration, the distortion coefficients
and one target rig configuration.

The **learned branch is a non-reproducible demonstration.** The training data,
losses, optimizer settings and checkpoint-selection splits behind the frozen
weights cannot be released, so its numbers record what one frozen configuration
did rather than something another group can regenerate and check. Freezing
weights at inference does not establish that evaluation frames were excluded
from training or model selection, and we do not claim it.

The L4-SE geometry-branch source is not in this snapshot either: the project
tree carries an empty placeholder, its documentation is in `code/docs/` and the
results it produced are in `evidence/`.

---

## Layout

```
code/       inference pipeline (GCR-NVS), rig configurations, environment files
evidence/   derived evaluation results: CSV, redacted JSON run reports
docs/       sensor, rig and data-scope documentation
tools/      contract property checker
```
