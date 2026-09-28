# Third-party components

The pipeline loads the external models below. **Each is used under its own
upstream licence, which is not restated here.** The project's internal manifest
(`code/docs/model_manifest.md`) records the requirement to capture each
licence, version and SHA256, but the licence column is not yet filled in, so
reproducing it here would mean asserting terms nobody has verified. Consult the
upstream project before redistributing or building on any of them.

| Component | Role in the pipeline | Upstream |
|---|---|---|
| DA3Metric-Large | RGB-aligned dense depth candidate | Depth Anything 3 |
| DINOv2 ViT-L/14 | Semantic features and correspondence | DINOv2 (Meta AI) |
| DINOv2 ViT-B/14 | Structure network's feature backbone | DINOv2 (Meta AI) |
| AnyUp | Feature upsampling, boundary recovery | AnyUp |
| Mask R-CNN R50-FPN | Dynamic-region test inside the gate | torchvision |
| Restormer-Small | Residual decoder | Restormer |
| NAFNet blocks | Detail branch | NAFNet |
| Stable Diffusion Inpainting | Optional residual-hole completion | Stability AI |

Two notes on what these are permitted to do, because the provenance contract
constrains them more tightly than their licences do:

- No learned module may write into the observed mask.
- No module whose output is labelled `generated` may contribute to the depth
  export, the SE(3) optimization or any reported geometric quantity. The
  optional diffusion component is a completion component only and is never
  geometric evidence.

The frozen checkpoints themselves are not redistributed. Their paths and
SHA256 digests are recorded in each run report under `evidence/`.
