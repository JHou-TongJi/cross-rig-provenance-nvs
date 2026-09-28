"""Diagnose source-semantic recovery before generative T0 hole completion."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage
import torch

from gcr_nvs.datasets.manifest import appearance_source_cameras, build_sequence_manifest
from gcr_nvs.geometry.calibration import load_calibrations, rectify_image
from gcr_nvs.models.dino_anyup_descriptor import FrozenDinoAnyUpDescriptor
from gcr_nvs.models.semantic_reference_completion import LocalSemanticReferenceRetriever


BLUE = np.asarray([25, 90, 235], dtype=np.uint8)


def _image(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"))


def _load_rectified_sources(bundle: Path, sequence: str, frame: int,
                            cameras: tuple[str, ...], size: tuple[int, int]) -> dict[str, np.ndarray]:
    sequence_root = bundle / "datasets" / sequence
    record = next(
        item for item in build_sequence_manifest(sequence_root, compute_quality=False)
        if item.frame_id == frame
    )
    calibrations = load_calibrations(
        sequence_root / record.camera_config,
        bundle / "camera_intric.yaml",
    )
    output = {}
    for camera in cameras:
        raw = _image(sequence_root / record.cameras[camera])
        rectified, _ = rectify_image(raw, calibrations[camera], output_size=size, alpha=0.0)
        output[camera] = rectified
    return output


def _pca_rgb(features: np.ndarray) -> np.ndarray:
    channels, height, width = features.shape
    flat = features.reshape(channels, -1).T
    sample = flat[::max(1, len(flat) // 12000)]
    mean = sample.mean(axis=0, keepdims=True)
    _, _, vectors = np.linalg.svd(sample - mean, full_matrices=False)
    projected = (flat - mean) @ vectors[:3].T
    lo = np.percentile(projected, 2, axis=0)
    hi = np.percentile(projected, 98, axis=0)
    projected = np.clip((projected - lo) / np.maximum(hi - lo, 1e-6), 0.0, 1.0)
    return np.rint(projected.reshape(height, width, 3) * 255).astype(np.uint8)


def _flow_retrieve(
    sources: list[np.ndarray],
    source_index: np.ndarray,
    offset_x: np.ndarray,
    offset_y: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    height, width = sources[0].shape[:2]
    low_h, low_w = offset_x.shape
    dx = cv2.resize(offset_x.astype(np.float32), (width, height), interpolation=cv2.INTER_LINEAR)
    dy = cv2.resize(offset_y.astype(np.float32), (width, height), interpolation=cv2.INTER_LINEAR)
    dx *= width / low_w
    dy *= height / low_h
    selected = cv2.resize(source_index.astype(np.int16), (width, height), interpolation=cv2.INTER_NEAREST)
    yy, xx = np.indices((height, width), dtype=np.float32)
    map_x, map_y = xx + dx, yy + dy
    inside = (map_x >= 0) & (map_x <= width - 1) & (map_y >= 0) & (map_y <= height - 1)
    result = np.zeros((height, width, 3), dtype=np.uint8)
    for view, source in enumerate(sources):
        sampled = cv2.remap(
            source, map_x, map_y, interpolation=cv2.INTER_LANCZOS4,
            borderMode=cv2.BORDER_CONSTANT,
        )
        choose = selected == view
        result[choose] = sampled[choose]
    return result, inside & (selected >= 0)


def _matched_source_depth(
    source_depths: list[np.ndarray],
    source_index: np.ndarray,
    offset_x: np.ndarray,
    offset_y: np.ndarray,
) -> np.ndarray:
    height, width = offset_x.shape
    yy, xx = np.indices((height, width), dtype=np.float32)
    map_x = xx + offset_x.astype(np.float32)
    map_y = yy + offset_y.astype(np.float32)
    result = np.full((height, width), np.nan, dtype=np.float32)
    for view, depth in enumerate(source_depths):
        sampled = cv2.remap(
            depth.astype(np.float32), map_x, map_y, interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT, borderValue=np.nan,
        )
        choose = source_index == view
        result[choose] = sampled[choose]
    return result


def _save_panel(output: Path, entries: list[tuple[str, np.ndarray]]) -> None:
    thumb = (960, 540)
    title_height = 42
    columns = 3
    rows = (len(entries) + columns - 1) // columns
    canvas = Image.new("RGB", (thumb[0] * columns, (thumb[1] + title_height) * rows), "white")
    font_path = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Medium.ttc")
    font = ImageFont.truetype(str(font_path), 25) if font_path.exists() else ImageFont.load_default()
    draw = ImageDraw.Draw(canvas)
    for index, (label, array) in enumerate(entries):
        image = Image.fromarray(array).resize(thumb, Image.Resampling.LANCZOS)
        x = index % columns * thumb[0]
        y = index // columns * (thumb[1] + title_height)
        draw.text((x + 12, y + 7), label, fill=(15, 20, 30), font=font)
        canvas.paste(image, (x, y + title_height))
    canvas.save(output, compress_level=2)


def main() -> None:
    default_bundle = Path("/media/2T_HD/GCR-NVS_DATA/t0_baseline_isolated_20260823")
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path, default=default_bundle)
    parser.add_argument("--audit", type=Path, default=default_bundle / "runs/t0_highres_source_audit_edge_all")
    parser.add_argument("--pose", default="mixed_20_50cm")
    parser.add_argument("--camera", default="CAM_FRONT_NARROW")
    parser.add_argument("--output", type=Path, default=default_bundle / "runs/t0_semantic_reference_diagnostic_v1")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--radius", type=int, default=12)
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--dino-model", choices=("dinov2_vitb14", "dinov2_vitl14"), default="dinov2_vitl14")
    parser.add_argument("--dino-input-width", type=int, default=1022)
    parser.add_argument("--dino-input-height", type=int, default=574)
    parser.add_argument("--retrieval-width", type=int, default=240)
    parser.add_argument("--retrieval-height", type=int, default=135)
    parser.add_argument("--descriptor-output-channels", type=int, default=64)
    parser.add_argument("--rgb-name", default="t0_highres_blue.png")
    parser.add_argument("--validity-name", default="validity.npy")
    parser.add_argument("--depth-name", default="depth.npy")
    args = parser.parse_args()

    summary = json.loads((args.audit / "summary.json").read_text(encoding="utf-8"))
    sequence, frame = summary["sequence"], int(summary["frame"])
    topology = appearance_source_cameras(args.camera, include_target=True)
    sample_root = args.audit / args.pose / args.camera
    t0 = _image(sample_root / args.rgb_name)
    validity = np.load(sample_root / args.validity_name).astype(bool)
    target_depth = np.load(sample_root / args.depth_name, mmap_mode="r").astype(np.float32)
    height, width = validity.shape
    sources_by_name = _load_rectified_sources(args.bundle, sequence, frame, topology, (width, height))
    sources = [sources_by_name[name] for name in topology]

    query_rgb = t0.copy()
    query_rgb[~validity] = sources[0][~validity]
    device = torch.device(args.device)
    descriptor = FrozenDinoAnyUpDescriptor(
        model_name=args.dino_model, output_channels=args.descriptor_output_channels,
        dino_input_size=(args.dino_input_height, args.dino_input_width),
        anyup_checkpoint=args.bundle / "outputs/checkpoints/anyup_multi_backbone.pth",
        anyup_root=args.bundle / "third_party/anyup",
    ).to(device).eval()
    retrieval_size = (args.retrieval_height, args.retrieval_width)
    query_tensor = torch.from_numpy(np.ascontiguousarray(query_rgb.transpose(2, 0, 1))).to(
        device=device, dtype=torch.float32,
    )[None] / 255.0
    query_features = descriptor(query_tensor, retrieval_size)
    reference_features = torch.stack(
        [descriptor(
            torch.from_numpy(np.ascontiguousarray(source.transpose(2, 0, 1))).to(
                device=device, dtype=torch.float32,
            )[None] / 255.0,
            retrieval_size,
        )[0] for source in sources], dim=0,
    )[None]
    low_sources = torch.from_numpy(np.stack([
        cv2.resize(source, (retrieval_size[1], retrieval_size[0]), interpolation=cv2.INTER_AREA)
        .transpose(2, 0, 1) for source in sources
    ])).to(device=device, dtype=torch.float32)[None] / 255.0
    low_hole = torch.from_numpy(
        cv2.resize((~validity).astype(np.uint8), (retrieval_size[1], retrieval_size[0]), interpolation=cv2.INTER_NEAREST)
    ).to(device=device)[None, None].bool()
    low_valid_np = ~low_hole[0, 0].cpu().numpy()
    target_depth_low = cv2.resize(
        target_depth, (retrieval_size[1], retrieval_size[0]), interpolation=cv2.INTER_NEAREST,
    )
    target_depth_low[~low_valid_np] = np.nan
    _, nearest_depth = ndimage.distance_transform_edt(~low_valid_np, return_indices=True)
    target_depth_condition = target_depth_low.copy()
    target_depth_condition[~low_valid_np] = target_depth_low[
        nearest_depth[0][~low_valid_np], nearest_depth[1][~low_valid_np]
    ]
    source_depths_low = [
        cv2.resize(
            np.load(args.audit / "source_sidecars" / name / "depth.npy", mmap_mode="r").astype(np.float32),
            (retrieval_size[1], retrieval_size[0]), interpolation=cv2.INTER_AREA,
        )
        for name in topology
    ]

    retriever = LocalSemanticReferenceRetriever(
        radius=args.radius, stride=args.stride, minimum_confidence=0.0,
    ).to(device).eval()
    with torch.inference_mode():
        match = retriever(query_features, reference_features, low_sources, low_hole)
        reverse_matches = []
        reverse_rgb = torch.zeros(
            (1, 1, 3, retrieval_size[0], retrieval_size[1]), device=device,
        )
        no_hole = torch.zeros_like(low_hole)
        for view in range(len(topology)):
            reverse_matches.append(retriever(
                reference_features[:, view], query_features[:, None], reverse_rgb, no_hole,
            ))
    low_valid = low_valid_np
    _, nearest = ndimage.distance_transform_edt(~low_valid, return_indices=True)
    nearest_y, nearest_x = nearest
    distance = ndimage.distance_transform_edt(~low_valid)

    source_index = match["source_index"][0, 0].cpu().numpy()
    offset_x = match["offset_x"][0, 0].cpu().numpy()
    offset_y = match["offset_y"][0, 0].cpu().numpy()
    similarity = match["similarity"][0, 0].cpu().numpy()
    cycle_error = np.full_like(similarity, np.inf, dtype=np.float32)
    grid_y, grid_x = np.indices(similarity.shape, dtype=np.float32)
    for view, reverse in enumerate(reverse_matches):
        choose = source_index == view
        if not choose.any():
            continue
        reverse_x = reverse["offset_x"][0, 0].cpu().numpy().astype(np.float32)
        reverse_y = reverse["offset_y"][0, 0].cpu().numpy().astype(np.float32)
        matched_x = grid_x + offset_x.astype(np.float32)
        matched_y = grid_y + offset_y.astype(np.float32)
        back_x = cv2.remap(reverse_x, matched_x, matched_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        back_y = cv2.remap(reverse_y, matched_x, matched_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        error = np.sqrt((offset_x + back_x) ** 2 + (offset_y + back_y) ** 2)
        cycle_error[choose] = error[choose]
    # The target descriptor inside a blue hole is only a proxy. Extend the
    # measured correspondence field from the nearest observed T0 pixel.
    for value in (source_index, offset_x, offset_y, similarity, cycle_error):
        value[~low_valid] = value[nearest_y[~low_valid], nearest_x[~low_valid]]

    matched_depth = _matched_source_depth(
        source_depths_low, source_index, offset_x, offset_y,
    )
    depth_log_error = np.abs(
        np.log(np.clip(matched_depth, 0.5, 250.0))
        - np.log(np.clip(target_depth_condition, 0.5, 250.0))
    )
    depth_log_error[~np.isfinite(matched_depth) | ~np.isfinite(target_depth_condition)] = np.inf

    retrieved_rgb, inside = _flow_retrieve(sources, source_index, offset_x, offset_y)
    similarity_hd = cv2.resize(similarity, (width, height), interpolation=cv2.INTER_LINEAR)
    cycle_error_hd = cv2.resize(cycle_error, (width, height), interpolation=cv2.INTER_LINEAR)
    depth_log_error_hd = cv2.resize(depth_log_error, (width, height), interpolation=cv2.INTER_LINEAR)
    distance_hd = cv2.resize(distance.astype(np.float32), (width, height), interpolation=cv2.INTER_LINEAR)
    max_distance = max(float(args.radius), 8.0)
    recoverable = (
        (~validity) & inside
        & (similarity_hd >= 0.68)
        & (cycle_error_hd <= 2.5)
        & (depth_log_error_hd <= 0.30)
        & (distance_hd <= max_distance)
    )
    residual = (~validity) & ~recoverable
    recovered = t0.copy()
    recovered[recoverable] = retrieved_rgb[recoverable]
    recovered[residual] = BLUE

    output = args.output / args.pose / args.camera
    output.mkdir(parents=True, exist_ok=True)
    Image.fromarray(retrieved_rgb).save(output / "retrieved_source_rgb.png")
    np.save(output / "input_validity.npy", validity.astype(bool))
    np.save(output / "input_depth.npy", target_depth.astype(np.float32))
    Image.fromarray(recovered).save(output / "t0_semantic_recovered_blue_residual.png")
    recoverable_u8 = np.uint8(recoverable) * 255
    residual_u8 = np.uint8(residual) * 255
    Image.fromarray(recoverable_u8, mode="L").save(output / "recoverable_mask.png")
    Image.fromarray(residual_u8, mode="L").save(output / "residual_true_hole_mask.png")
    target_depth_hd = cv2.resize(target_depth_condition, (width, height), interpolation=cv2.INTER_LINEAR)
    target_depth_hd[validity] = target_depth[validity]
    np.save(output / "target_depth_condition.npy", target_depth_hd.astype(np.float32))
    # Save low-resolution diagnostics for the correspondence-head training
    # pilot. These are geometry/semantic signals only; target RGB is kept as
    # the external supervision source and never enters the condition tensor.
    low_distance = cv2.resize(distance.astype(np.float32), (retrieval_size[1], retrieval_size[0]), interpolation=cv2.INTER_LINEAR)
    low_inside = cv2.resize(inside.astype(np.float32), (retrieval_size[1], retrieval_size[0]), interpolation=cv2.INTER_NEAREST)
    low_depth_error = np.nan_to_num(depth_log_error, nan=1.0, posinf=1.0).astype(np.float32)
    low_hole_np = (~low_valid).astype(np.float32)
    low_features = np.stack([
        similarity.astype(np.float32),
        cycle_error.astype(np.float32) / 6.0,
        low_depth_error,
        low_distance / max(max_distance, 1.0),
        low_inside,
        low_hole_np,
        np.ones_like(similarity, dtype=np.float32),
    ], axis=-1)
    np.save(output / "correspondence_features.npy", low_features.astype(np.float32))
    target_rgb_low = cv2.resize(sources[0], (retrieval_size[1], retrieval_size[0]), interpolation=cv2.INTER_AREA)
    retrieved_low = cv2.resize(retrieved_rgb, (retrieval_size[1], retrieval_size[0]), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    oracle_error = np.abs(target_rgb_low.astype(np.float32) / 255.0 - retrieved_low).mean(axis=2)
    oracle = (oracle_error <= 0.12) & (low_hole_np > 0.5) & np.isfinite(low_depth_error) & (low_depth_error <= 0.30)
    np.save(output / "correspondence_oracle.npy", oracle.astype(np.uint8))
    np.save(output / "correspondence_oracle_error.npy", oracle_error.astype(np.float32))
    similarity_vis = cv2.applyColorMap(np.uint8(np.clip(similarity_hd, 0, 1) * 255), cv2.COLORMAP_TURBO)[..., ::-1]
    cycle_vis = cv2.applyColorMap(
        np.uint8(np.clip(cycle_error_hd / 6.0, 0, 1) * 255), cv2.COLORMAP_TURBO,
    )[..., ::-1]
    depth_vis = cv2.applyColorMap(
        np.uint8(np.clip(np.nan_to_num(depth_log_error_hd, nan=0.6, posinf=0.6) / 0.6, 0, 1) * 255), cv2.COLORMAP_TURBO,
    )[..., ::-1]
    source_hd = cv2.resize(source_index.astype(np.float32), (width, height), interpolation=cv2.INTER_NEAREST)
    source_vis = cv2.applyColorMap(
        np.uint8(np.clip((source_hd + 1) / (len(sources) + 1), 0, 1) * 255), cv2.COLORMAP_VIRIDIS,
    )[..., ::-1]
    query_pca = cv2.resize(_pca_rgb(query_features[0].cpu().numpy()), (width, height), interpolation=cv2.INTER_NEAREST)
    _save_panel(output / "semantic_reference_diagnostic_panel.png", [
        ("T0 原始蓝洞", t0),
        ("主源图 RGB", sources[0]),
        (f"{args.dino_model} + AnyUP 语义", query_pca),
        ("语义匹配相似度", similarity_vis),
        ("往返匹配误差", cycle_vis),
        ("源/目标深度误差", depth_vis),
        ("源图真实 RGB 检索", retrieved_rgb),
        ("可找回区域", np.repeat(recoverable_u8[..., None], 3, axis=2)),
        ("仍需 DeepFill v2", np.repeat(residual_u8[..., None], 3, axis=2)),
        ("语义找回后 T0", recovered),
    ])
    report = {
        "sequence": sequence,
        "frame": frame,
        "pose": args.pose,
        "camera": args.camera,
        "source_topology": list(topology),
        "resolution": [width, height],
        "retrieval_resolution": [retrieval_size[1], retrieval_size[0]],
        "input_rgb": args.rgb_name,
        "input_validity": args.validity_name,
        "input_depth": args.depth_name,
        "dino": args.dino_model,
        "dino_input_size": [args.dino_input_width, args.dino_input_height],
        "anyup": True,
        "original_hole_fraction": float((~validity).mean()),
        "recoverable_fraction_of_image": float(recoverable.mean()),
        "recoverable_fraction_of_hole": float(recoverable.sum() / max((~validity).sum(), 1)),
        "residual_fraction_of_image": float(residual.mean()),
        "matching_contract": {
            "minimum_cosine_similarity": 0.68,
            "maximum_cycle_error_at_retrieval_grid": 2.5,
            "maximum_log_depth_error": 0.30,
            "maximum_extrapolation_distance_at_retrieval_grid": max_distance,
        },
        "observed_pixels_modified": int(np.any(recovered[validity] != t0[validity], axis=1).sum()),
        "note": "DeepFill v2 must receive only residual_true_hole_mask.png",
        "correspondence_training_artifacts": {
            "features": "correspondence_features.npy",
            "oracle": "correspondence_oracle.npy",
            "oracle_definition": "target-camera rectified RGB versus retrieved source RGB mean absolute error <= 0.12, plus depth gate",
        },
    }
    (output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(output), **report}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
