# Deployment and reproduction

Three things in this repository run, and they have very different requirements.
Read the table first and skip to whichever row you need.

| What you want to run | Needs a GPU | Needs the road data | Needs model weights |
|---|---|---|---|
| Contract property checker | no | no | no |
| Manuscript build and the five audits | no | no | no |
| The GCR-NVS pipeline itself | yes | yes | yes |

The first two run on a clean clone in minutes. The third cannot be run from
this repository alone, for reasons set out at the end.

---

## 1. Contract property checker

The only part that needs nothing but Python. It checks the provenance
contract's six invariants against a reference implementation written from the
article, with negative controls that break each rule and require the matching
invariant to fail.

```bash
cd paper
python contract_properties.py
```

Expected output ends with:

```
PASS: 6 properties hold over 20000 frames; all 4 injected violations were detected
```

Python 3.9 or newer. No third-party packages.

---

## 2. Manuscript build and audits

Rebuilds the submission document from the plain-text sources and re-derives
every number in it from the evaluation data.

```bash
python -m venv .venv && . .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install python-docx==1.1.2 latex2mathml pymupdf pillow numpy matplotlib

cd paper
python assemble.py              # src_*.txt -> manuscript.txt
python build_wevj_docx.py       # render into the MDPI template
```

`build_wevj_docx.py` produces the DOCX with native Word equations and SEQ/REF
fields. The field codes are written but not evaluated, so cross-references and
page numbers are placeholders until Word resolves them:

```powershell
pwsh -File finalize.ps1         # Windows + Microsoft Word required
```

This step drives Word through COM to update fields, repaginate and export the
PDF. There is no substitute: `python-docx` writes field codes and cannot
evaluate them. Without Word you still get a valid DOCX, with unresolved
numbering.

### The audits

Order matters, because three of them read build products rather than sources.
`manuscript.txt` is generated, not stored, so `assemble.py` comes first:

```bash
cd paper
python assemble.py               # required first: writes manuscript.txt

python verify_manuscript.py      # cross-references, 57 numeric claims, citation order
python audit_arithmetic.py       # re-derives every relational claim from the CSVs
python audit_redundancy.py       # sentence- and block-level duplication
python audit_referents.py        # prints each citing sentence beside the caption it names

python build_wevj_docx.py        # required before the next one
python audit_docx.py             # field errors, glyphs, stray paths, document properties

pwsh -File finalize.ps1          # required before the last one (needs Word)
python audit_layout.py cross_vehicle_nvs_wevj_submission.pdf
```

The four after `assemble.py` need nothing but the sources. `audit_docx.py`
needs a built DOCX and `audit_layout.py` a built PDF, so they sit after the
steps that produce them.
`audit_referents.py` reports rather than judges: it prints pairs for a human to
read, because deciding whether a sentence describes the object it cites needs
someone who knows what the sentence claims.

No number in the manuscript is hand-typed. `audit_arithmetic.py` is what
enforces that, and it fails loudly rather than printing a reassuring summary.

---

## 3. The GCR-NVS pipeline

`code/gcr-nvs-inference-only-20260826/` holds the inference implementation,
configurations and environment files. The released configuration is
`configs/releases/t0_uniview_a_final.yaml`.

```bash
pip install -r code/requirements.gcr-core.txt      # torch 2.4.0 / CUDA 12.4
pip install -r code/requirements.gcr-optional.txt  # optional visual components
```

Tested under Python 3.11 with CUDA 12.4 on Linux; the CUDA runtime arrives with
the PyTorch wheels. `code/model_manifest.md` lists every external model, its
version, licence and SHA256.

### What you cannot do from this repository

Running the pipeline end to end needs three things that are not here, and the
article says so rather than implying otherwise:

- **The road-collection sequence.** Used under a challenge data-use agreement
  that forbids redistribution. Requests go to the data provider.
- **The source vehicle's manufacturer calibration.** Withheld for the same
  reason; the rig pose blocks inside the JSON run reports are redacted field by
  field, so every coverage number the article cites survives while the matrices
  behind them do not.
- **The frozen checkpoints.** Referenced by controlled path, about 608 MB,
  available from the corresponding author on reasonable request subject to the
  third-party licences in `code/model_manifest.md`.

The split that follows is stated in Section 8.1 of the article and is worth
repeating here. The **geometry branch** carries no learned component, so its
results reproduce in principle given the sequence, the calibration, the
distortion coefficients and one target rig configuration. The **learned branch
is a non-reproducible demonstration**: the training data, losses, optimizer
settings and checkpoint-selection splits behind the frozen weights cannot be
released, so those tables record what one frozen configuration did rather than
something another group can regenerate and check.

The L4-SE geometry-branch source is not in this snapshot either; the project
tree carries an empty placeholder for it, its documentation is in `code/docs/`
and the results it produced are in `results/`.

---

## Directory layout

```
paper/    manuscript sources, build scripts, audits, figures, submission PDF
code/     the inference pipeline, configurations, environment files
results/  derived evaluation results: CSV, redacted JSON run reports, summaries
docs/     sensor, rig and data-scope documentation
```

`assemble_repo.py` in the source project applies the redaction rules and prints
everything it withholds, so the boundary between what is here and what is not
is reproducible rather than a one-time judgement.
