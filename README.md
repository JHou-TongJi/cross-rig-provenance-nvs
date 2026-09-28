# A Per-Pixel Provenance Contract and Evidence Profile

Code, derived results and manuscript sources for the article

**A Per-Pixel Provenance Contract and Evidence Profile for Cross-Rig Camera and LiDAR Reprojection: Specification and Proxy Audit**

submitted to *World Electric Vehicle Journal* (MDPI).

## What the work is about

A recorded multi-camera corpus stops matching its vehicle as soon as the sensor
rig is re-specified — a camera moves, gets a different lens, or the mounting
changes. Reprojecting the corpus into the new rig is the obvious response, and
the question this article addresses is how anyone would know whether the result
may be trusted.

The answer here has two parts. A **provenance contract** assigns every delivered
pixel to exactly one of three classes — `observed` where a calibrated
correspondence supports it, `gate-recovered` where geometry can be shown to
support it, `generated` otherwise — and enforces that partition in the
compositing code rather than documenting it afterwards. Observed pixels are
locked byte-exactly on the rectified internal raster the lock is defined over, and no
visual-only module may write depth, pose or any reported geometric quantity. The
exported image is redistorted by interpolation, so byte equality is claimed before
that resampling and not after it.

An **evidence profile** then replaces the photometric score that is undefined in
this regime, because the target platform was never driven and so no target-view
reference image exists. The profile reports observed coverage, gate acceptance
under an explicit cycle test, generated fraction, locked-pixel compliance,
per-camera risk and temporal spread, each with a defined support and a stated
measurement status.

Two findings are not specific to this pipeline:

- A seven-camera average conceals a roughly fivefold concentration of
  vehicle-region risk in one camera (24.23% against a 4.94% mean). Two optical
  mechanisms are consistent with it and are reported as hypotheses, because one
  sequence cannot separate optics from the scene content that camera faces.
  Note that the LiDAR figure is per pixel: per solid angle the same camera is the
  densest on the rig, not the sparsest.
- A delivered video can be tested against its declared construction from the
  pixels alone; the released sequence carries 2 Hz of measurement at 120 fps.

The article is explicit about what it does **not** establish: it reports no
held-out geometric error and no downstream retention, so it does not demonstrate
that a reprojected corpus is fit for training or validating a perception stack.
It contributes an auditable route to that determination.

## Layout

```
paper/      manuscript sources and the scripts that build the submission DOCX
code/       the inference pipeline (GCR-NVS), configs and environment files
results/    derived evaluation results: CSV, JSON run reports, summaries
docs/       sensor, rig and data-scope documentation
```

## What is not here, and why

The reference sequence is used under a challenge data-use agreement. The
article's Data Availability Statement commits to not redistributing it, and this
repository holds to that:

| Withheld | Reason |
|---|---|
| Raw RGB imagery and LiDAR point clouds | Data-use agreement; not redistributable |
| `camera_config.json`, `camera_intric.yaml` | Manufacturer calibration of the source vehicle |
| `target_camera_module_7v.json` | Re-packages the same supplied extrinsics under another name |
| Model weights (`*.pt`) | Referenced by controlled path; ~608 MB |
| Rig pose blocks inside JSON run reports | Derived from the withheld calibration; redacted in place |

Redaction was done by field, not by dropping files: every coverage number the
article cites survives in `results/`, and the pose matrices those numbers were
computed with read
`"withheld: derived from source calibration not redistributed"`.

Two further gaps are worth stating plainly rather than leaving to be discovered:

- **The L4-SE geometry-branch source is not in this snapshot.** The project tree
  carries an empty placeholder for it. The documentation in `code/docs/` describes
  that branch, and the results it produced are in `results/`, but the code is not
  here.
- **Camera focal lengths are published.** Table 1 of the article lists them per
  camera, so the audit scripts carry them; nothing beyond what the article
  already prints is exposed.

`assemble_repo.py` in the source project applies these rules and prints
everything it withholds, so the boundary is reproducible rather than a one-time
judgement.

## Running this

See **[DEPLOYMENT.md](DEPLOYMENT.md)** for setup, dependencies and what each part needs. In short: the contract property checker needs only Python, the manuscript build and audits need a few packages (and Word for the final repagination), and the pipeline itself needs a GPU plus data and weights that are not redistributable.

## Rebuilding the manuscript

The manuscript is not written in the DOCX. It is assembled from plain-text
sources into the MDPI template, so that numbers and cross-references can be
checked mechanically.

```bash
cd paper
python assemble.py            # concatenate src_*.txt -> manuscript.txt
python build_wevj_docx.py     # render into the MDPI template
pwsh -File finalize.ps1       # update fields, repaginate, export PDF (needs Word)
```

`finalize.ps1` drives Word through COM: `python-docx` writes the SEQ/REF field
codes but cannot evaluate them, so Word has to resolve the numbering and
repaginate.

### Checks

These run over the manuscript; the five audits pass on the submitted version, and
`contract_properties.py` reports its own result.

| Script | What it checks |
|---|---|
| `verify_manuscript.py` | Cross-references, 57 numeric claims, citation order |
| `audit_arithmetic.py` | Re-derives every relational claim from the raw data |
| `audit_redundancy.py` | Sentence- and block-level duplication |
| `audit_docx.py` | Field errors, glyph damage, stray paths, literal LaTeX |
| `audit_layout.py` | Caption sequence, empty headings, figure-text geometry |
| `renumber.py` | Recomputes figure and table numbers from document order |
| `audit_referents.py` | Prints each citing sentence beside the caption it names |
| `contract_properties.py` | Checks the contract's six invariants, with negative controls |

No result value in the manuscript is hand-typed: every number is derived by
script from the raw evaluation data, which is what `audit_arithmetic.py` exists
to enforce.

## Reproducing the results

Full reproduction needs the raw sequence and the model weights, neither of which
is redistributable here. What the repository supports without them:

- reading every derived result in `results/`;
- rebuilding the manuscript and re-running the audits that do not need the raw
  sequence;
- reading the pipeline source and configuration in `code/`.

The split is worth stating plainly. The geometry branch carries no learned
component, so its results reproduce in principle given the source sequence, the
calibration, the distortion coefficients and one target rig configuration. **The
learned-branch results are a non-reproducible demonstration**: the training data,
losses, optimizer settings and checkpoint-selection splits behind the frozen
weights cannot be released, so those tables record what one frozen configuration
did rather than something another group can regenerate and check.

`contract_properties.py` is the exception that needs nothing at all. It checks the
contract's invariants against a reference implementation written from the article,
with negative controls that break each rule and require the matching property to
fail. It runs in seconds on a clean clone.

## Citation

The article is under review; citation details will be added on acceptance.

## Licence

No licence has been chosen yet. Until one is added, default copyright applies
and no reuse rights are granted. Note that the code depends on third-party
models (DINOv2, Depth Anything 3, AnyUp) under their own licences, recorded in
`code/model_manifest.md`.
