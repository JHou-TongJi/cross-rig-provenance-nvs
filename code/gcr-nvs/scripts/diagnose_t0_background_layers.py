"""Diagnose background surfaces exposed by a target SE(3) perturbation.

This is deliberately a diagnostic: it does not alter T0 outputs or invoke a
generator. It keeps the nearest and second-nearest projected source surfaces,
then asks whether residual target holes have a real RGB source behind the
currently rendered foreground.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from gcr_nvs.datasets.manifest import appearance_source_cameras, build_sequence_manifest
from gcr_nvs.geometry.calibration import load_calibrations, rectified_calibration, rectify_image
from gcr_nvs.geometry.rig import load_target_rig, resolve_target_calibration
from gcr_nvs.rendering.dense_depth_reprojection import _project


POSES = {
    "mixed_5_10cm": (0.06, -0.08, 0.05),
    "mixed_10_20cm": (-0.14, 0.12, 0.18),
    "mixed_20_50cm": (0.32, -0.24, 0.42),
}
CAMERAS = (
    "CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT", "CAM_FRONT_LEFT",
    "CAM_FRONT_NARROW", "CAM_FRONT_RIGHT", "CAM_FRONT_WIDE",
)


def _perturb(calibration, translation):
    camera_to_world = np.linalg.inv(calibration.external)
    camera_to_world[:3, 3] += np.asarray(translation, np.float64)
    return type(calibration)(
        calibration.name, calibration.intrinsic.copy(), calibration.distortion.copy(),
        np.linalg.inv(camera_to_world), calibration.width, calibration.height,
    )


def _font(size: int):
    path = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Medium.ttc")
    return ImageFont.truetype(str(path), size) if path.exists() else ImageFont.load_default()


def _resize_rgb(path: Path, size: tuple[int, int]) -> np.ndarray:
    image = np.asarray(Image.open(path).convert("RGB"))
    return cv2.resize(image, size, interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0


def _resize_depth(path: Path, size: tuple[int, int]) -> np.ndarray:
    depth = np.load(path).astype(np.float32)
    return cv2.resize(depth, size, interpolation=cv2.INTER_NEAREST)


def _resize_semantic(path: Path, size: tuple[int, int]) -> np.ndarray:
    """Return HWC semantic features; PCA images are never used as features."""
    features = np.load(path, mmap_mode="r").astype(np.float32)
    if features.ndim != 3:
        raise ValueError(f"Expected CHW semantic features, got {features.shape} from {path}")
    if features.shape[0] <= features.shape[-1]:
        features = np.transpose(features, (1, 2, 0))
    return np.stack(
        [cv2.resize(features[..., channel], size, interpolation=cv2.INTER_LINEAR)
         for channel in range(features.shape[-1])],
        axis=-1,
    )


def _scale_calibration(calibration, size: tuple[int, int]):
    width, height = size
    result = rectified_calibration(calibration, output_size=size, alpha=0.0)
    return result


def _project_source(
    rgb: np.ndarray,
    depth: np.ndarray,
    semantic: np.ndarray,
    source_calibration,
    target_calibration,
    source_index: int,
):
    height, width = depth.shape
    yy, xx = np.indices(depth.shape, dtype=np.float32)
    z = depth
    valid = np.isfinite(z) & (z > 1e-3)
    x = (xx - source_calibration.intrinsic[0, 2]) / source_calibration.intrinsic[0, 0] * z
    y = (yy - source_calibration.intrinsic[1, 2]) / source_calibration.intrinsic[1, 1] * z
    points = np.stack([x, y, z], axis=-1).reshape(-1, 3)
    world = (
        np.linalg.inv(source_calibration.external)
        @ np.c_[points, np.ones(len(points), dtype=np.float32)].T
    ).T[:, :3]
    uv, target_z, projected = _project(world, target_calibration)
    valid = valid.reshape(-1) & projected & np.isfinite(target_z)
    candidates = np.flatnonzero(valid)
    if not len(candidates):
        return None
    pixels = np.rint(uv[candidates]).astype(np.int64)
    inside = (
        (pixels[:, 0] >= 0) & (pixels[:, 0] < target_calibration.width)
        & (pixels[:, 1] >= 0) & (pixels[:, 1] < target_calibration.height)
    )
    if not inside.any():
        return None
    candidates = candidates[inside]
    pixels = pixels[inside]
    target_z = target_z[candidates]
    flat = pixels[:, 1] * target_calibration.width + pixels[:, 0]
    return {
        "flat": flat,
        "z": target_z.astype(np.float32),
        "source_xy": candidates,
        "rgb": rgb.reshape(-1, 3)[candidates],
        "semantic": semantic.reshape(-1, semantic.shape[-1])[candidates],
        "source": np.full(len(candidates), source_index, dtype=np.int16),
        "world": world[candidates].astype(np.float32),
        "source_calibration": source_calibration,
    }


def _triple_reprojection_filter(
    candidates: list[dict],
    first_depth: np.ndarray,
    target_calibration,
    shape: tuple[int, int],
    pixel_tolerance: float = 1.5,
) -> tuple[list[dict], np.ndarray]:
    """Filter DA3 surfaces that cannot survive target z-buffer round-trip.

    The first projection is source -> target. The second projection uses the
    target z-buffer winner to reconstruct a world point and reprojects it back
    to the source camera. A candidate is retained only when it is the target
    winner and returns close to its original source pixel. This is the
    visibility part of UniWorld-View's triple reprojection; it does not invent
    RGB and it does not treat the calibrated structure depth as appearance.
    """
    height, width = shape
    filtered: list[dict] = []
    target_visible = np.zeros((height, width), dtype=bool)
    target_inverse = np.linalg.inv(target_calibration.external)
    for item in candidates:
        flat = item["flat"]
        z = item["z"]
        first = first_depth.reshape(-1)[flat]
        front = np.isfinite(first) & (np.abs(z - first) <= np.maximum(0.02, first * 0.008))
        if not front.any():
            continue
        selected_flat = flat[front]
        selected_z = first[front]
        uv = np.stack(
            [selected_flat % width, selected_flat // width], axis=-1,
        ).astype(np.float32)
        x = (uv[:, 0] - target_calibration.intrinsic[0, 2]) / target_calibration.intrinsic[0, 0] * selected_z
        y = (uv[:, 1] - target_calibration.intrinsic[1, 2]) / target_calibration.intrinsic[1, 1] * selected_z
        target_world = (
            target_inverse @ np.c_[x, y, selected_z, np.ones(len(selected_z), np.float32)].T
        ).T[:, :3]
        source_camera = (
            item["source_calibration"].external
            @ np.c_[target_world, np.ones(len(target_world), np.float32)].T
        ).T[:, :3]
        source_uv = np.full((len(source_camera), 2), np.nan, np.float32)
        valid = np.isfinite(source_camera).all(axis=1) & (source_camera[:, 2] > 1e-4)
        source_uv[valid] = (
            source_camera[valid, :2] / source_camera[valid, 2:3]
            @ item["source_calibration"].intrinsic[:2, :2].T
            + item["source_calibration"].intrinsic[:2, 2]
        )
        source_flat = item["source_xy"][front]
        source_uv_expected = np.stack(
            [source_flat % item["source_calibration"].width, source_flat // item["source_calibration"].width], axis=-1,
        ).astype(np.float32)
        error = np.linalg.norm(source_uv - source_uv_expected, axis=1)
        keep = valid & np.isfinite(error) & (error <= pixel_tolerance)
        if not keep.any():
            continue
        item_copy = dict(item)
        for key in ("flat", "z", "source_xy", "rgb", "semantic", "source", "world"):
            item_copy[key] = item[key][front][keep]
        filtered.append(item_copy)
        target_visible[item_copy["flat"] // width, item_copy["flat"] % width] = True
    return filtered, target_visible


def _layer_maps(candidates, shape, margin_m: float = 0.05):
    height, width = shape
    count = height * width
    first_depth = np.full(count, np.inf, np.float32)
    for item in candidates:
        np.minimum.at(first_depth, item["flat"], item["z"])
    second_depth = np.full(count, np.inf, np.float32)
    for item in candidates:
        flat, z = item["flat"], item["z"]
        first = first_depth[flat]
        behind = z > first + np.maximum(margin_m, first * 0.015)
        if behind.any():
            np.minimum.at(second_depth, flat[behind], z[behind])

    semantic_channels = candidates[0]["semantic"].shape[-1] if candidates else 0

    def _attributes(depth, behind_first):
        rgb = np.zeros((count, 3), np.float32)
        semantic = np.zeros((count, semantic_channels), np.float32)
        provenance = np.full(count, -1, np.int16)
        for item in candidates:
            flat, z = item["flat"], item["z"]
            reference = depth[flat]
            choose = np.isfinite(reference) & (np.abs(z - reference) <= np.maximum(0.02, reference * 0.008))
            if behind_first is not None:
                choose &= z > first_depth[flat] + np.maximum(margin_m, first_depth[flat] * 0.015)
            if choose.any():
                selected = flat[choose]
                rgb[selected] = item["rgb"][choose]
                semantic[selected] = item["semantic"][choose]
                provenance[selected] = item["source"][choose]
        return rgb, semantic, provenance

    first_rgb, first_semantic, first_provenance = _attributes(first_depth, None)
    second_rgb, second_semantic, second_provenance = _attributes(second_depth, first_depth)
    semantic_shape = (height, width, semantic_channels)
    return {
        "first_depth": first_depth.reshape(height, width),
        "second_depth": second_depth.reshape(height, width),
        "first_rgb": first_rgb.reshape(height, width, 3),
        "second_rgb": second_rgb.reshape(height, width, 3),
        "first_semantic": first_semantic.reshape(semantic_shape),
        "second_semantic": second_semantic.reshape(semantic_shape),
        "first_provenance": first_provenance.reshape(height, width),
        "second_provenance": second_provenance.reshape(height, width),
    }


def _depth_vis(depth: np.ndarray) -> np.ndarray:
    valid = np.isfinite(depth) & (depth > 1e-3)
    value = np.zeros(depth.shape, np.uint8)
    if valid.any():
        log_depth = np.log(np.clip(depth[valid], 0.5, 160.0))
        value[valid] = np.uint8(np.clip((log_depth - np.log(0.5)) / np.log(320.0) * 255, 0, 255))
    return cv2.applyColorMap(255 - value, cv2.COLORMAP_TURBO)[..., ::-1]


def _semantic_vis(features: np.ndarray) -> np.ndarray:
    """PCA-like visualization for a feature tensor; features remain lossless in .npy."""
    if features.ndim != 3 or features.shape[-1] == 0:
        return np.zeros((*features.shape[:2], 3), np.float32)
    flat = features.reshape(-1, features.shape[-1]).astype(np.float32)
    valid = np.isfinite(flat).all(axis=1)
    if valid.sum() < 3:
        return np.zeros((*features.shape[:2], 3), np.float32)
    sample = flat[valid]
    if len(sample) > 50000:
        sample = sample[:: max(1, len(sample) // 50000)]
    centered = sample - sample.mean(0, keepdims=True)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    basis = vt[:3].T
    values = (flat - sample.mean(0, keepdims=True)) @ basis
    values = values.reshape(*features.shape[:2], 3)
    lo = np.nanpercentile(values, 2, axis=(0, 1), keepdims=True)
    hi = np.nanpercentile(values, 98, axis=(0, 1), keepdims=True)
    return np.clip((values - lo) / np.maximum(hi - lo, 1e-6), 0, 1).astype(np.float32)


def _save_sheet(path: Path, entries: list[tuple[str, np.ndarray]]) -> None:
    cell = (640, 360)
    title = 38
    canvas = Image.new("RGB", (cell[0] * 4, (cell[1] + title) * ((len(entries) + 3) // 4)), "white")
    draw = ImageDraw.Draw(canvas)
    for index, (label, image) in enumerate(entries):
        x = index % 4 * cell[0]
        y = index // 4 * (cell[1] + title)
        draw.text((x + 8, y + 6), label, fill=(15, 20, 30), font=_font(17))
        canvas.paste(Image.fromarray(np.uint8(np.clip(image, 0, 1) * 255) if image.dtype != np.uint8 else image).resize(cell, Image.Resampling.LANCZOS), (x, y + title))
    canvas.save(path, quality=95, subsampling=0)


def main() -> None:
    project = Path(__file__).resolve().parents[1]
    default_bundle = Path("/media/2T_HD/GCR-NVS_DATA/t0_baseline_isolated_20260823")
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path, default=default_bundle)
    parser.add_argument("--audit-root", type=Path, default=default_bundle / "runs/t0_highres_source_audit_edge_all")
    parser.add_argument("--repair-root", type=Path, default=default_bundle / "runs/t0_target_surface_2px_repair_all_20260824")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pose", choices=(*POSES, "all"), default="all")
    parser.add_argument("--camera", choices=(*CAMERAS, "all"), default="all")
    parser.add_argument("--scale", type=float, default=0.25)
    parser.add_argument("--margin-m", type=float, default=0.05)
    parser.add_argument(
        "--depth-layer", choices=("da3", "structure"), default="da3",
        help="Depth used to build RGB-aligned source surfaces. DA3 is the default; structure is diagnostic only.",
    )
    parser.add_argument(
        "--triple-reprojection", action=argparse.BooleanOptionalAction, default=True,
        help="Apply UniWorld-style source->target->source visibility filtering.",
    )
    parser.add_argument("--triple-pixel-tolerance", type=float, default=1.5)
    parser.add_argument(
        "--all-source-cameras", action="store_true",
        help="Use all seven cameras for background evidence diagnosis, ignoring appearance topology.",
    )
    parser.add_argument("--target-rig", type=Path,
        help="Diagnose a target_rig output produced by render_t0_highres_source_audit.")
    parser.add_argument("--source-root", type=Path,
        help="Raw sequence root when --target-rig is used.")
    parser.add_argument("--target-cameras", nargs="+", choices=CAMERAS,
        help="Restrict target cameras while keeping topology source cameras available.")
    parser.add_argument("--distortion", type=Path,
        help="Brown distortion YAML. Required for a rectified RGB/calibration contract.")
    args = parser.parse_args()
    summary = json.loads((args.audit_root / "summary.json").read_text(encoding="utf-8"))
    sequence, frame = summary["sequence"], int(summary["frame"])
    sequence_root = (args.source_root / sequence) if args.source_root else (args.bundle / sequence)
    record = next(r for r in build_sequence_manifest(sequence_root, compute_quality=False) if r.frame_id == frame)
    distortion = args.distortion or (args.bundle / "camera_intric.yaml")
    raw_cals = load_calibrations(sequence_root / record.camera_config, distortion)
    output_group = args.audit_root / "target_rig" if args.target_rig else args.repair_root / ("mixed_5_10cm" if args.pose == "all" else args.pose)
    high_size = tuple(Image.open(output_group / CAMERAS[0] / "source_native_rectified.png").size)
    size = (int(round(high_size[0] * args.scale)), int(round(high_size[1] * args.scale)))
    rectified = {name: _scale_calibration(cal, size) for name, cal in raw_cals.items()}
    rig = load_target_rig(args.target_rig) if args.target_rig else None
    poses = ["target_rig"] if rig else (list(POSES) if args.pose == "all" else [args.pose])
    targets = [spec.name for spec in rig.cameras] if rig and args.camera == "all" else ([args.camera] if args.camera != "all" else list(CAMERAS))
    if args.target_cameras:
        targets = list(args.target_cameras)
    source_depth_root = args.audit_root / "source_sidecars" if rig else args.repair_root / "source_sidecars"
    args.output.mkdir(parents=True, exist_ok=True)
    report = {}
    for pose in poses:
        report[pose] = {}
        translation = POSES.get(pose, (0.0, 0.0, 0.0))
        for target_name in targets:
            target_folder = output_group / target_name
            if rig:
                target_cal = _scale_calibration(
                    resolve_target_calibration(rig.camera(target_name), raw_cals), size,
                )
            else:
                target_cal = _perturb(rectified[target_name], translation)
            hole_hd = ~np.load(target_folder / "validity_after_2px_surface_repair.npy").astype(bool)
            # Preserve the original invalid topology when making a diagnostic
            # preview. INTER_AREA with ``> 0`` expands one invalid HD pixel into
            # a whole low-resolution block and creates artificial seams.
            hole = cv2.resize(hole_hd.astype(np.uint8), size, interpolation=cv2.INTER_NEAREST).astype(bool)
            candidates = []
            source_names = (
                CAMERAS if args.all_source_cameras
                else appearance_source_cameras(
                    rig.camera(target_name).inherit_source_camera if rig else target_name,
                    include_target=True,
                )
            )
            for source_index, source_name in enumerate(source_names):
                source_folder = output_group / source_name
                if not (source_folder / "source_native_rectified.png").exists():
                    source_folder = args.audit_root / "source_sidecars" / source_name
                source_rgb_path = source_folder / "source_native_rectified.png"
                if source_rgb_path.exists():
                    rgb = _resize_rgb(source_rgb_path, size)
                else:
                    raw_image = np.asarray(Image.open(sequence_root / record.cameras[source_name]).convert("RGB"))
                    rgb_rectified, _ = rectify_image(raw_image, raw_cals[source_name], output_size=size, alpha=0.0)
                    rgb = rgb_rectified.astype(np.float32) / 255.0
                depth_name = "da3_depth_before_structure.npy" if args.depth_layer == "da3" else "depth.npy"
                depth = _resize_depth(source_depth_root / source_name / depth_name, size)
                semantic_path = source_depth_root / source_name / "dino_anyup_semantic_320x180.npy"
                semantic = _resize_semantic(semantic_path, size) if semantic_path.exists() else np.zeros((*rgb.shape[:2], 0), np.float32)
                item = _project_source(rgb, depth, semantic, rectified[source_name], target_cal, source_index)
                if item is not None:
                    candidates.append(item)
            triple_visibility = np.zeros((size[1], size[0]), dtype=bool)
            if args.triple_reprojection and candidates:
                pre_layers = _layer_maps(candidates, (size[1], size[0]), args.margin_m)
                candidates, triple_visibility = _triple_reprojection_filter(
                    candidates, pre_layers["first_depth"], target_cal, (size[1], size[0]), args.triple_pixel_tolerance,
                )
            layers = _layer_maps(candidates, (size[1], size[0]), args.margin_m)
            support = hole & np.isfinite(layers["first_depth"])
            behind = hole & np.isfinite(layers["second_depth"])
            # For a residual hole, the nearest observed source surface is the
            # exposed background candidate; the second layer is an extra check
            # that a distinct occluding surface exists in the source set.
            background = support & (layers["first_provenance"] >= 0)
            true_disocclusion = hole & ~background
            out = args.output / pose / target_name
            out.mkdir(parents=True, exist_ok=True)
            t0_rgb = _resize_rgb(target_folder / "t0_surface_2px_repaired_blue.png", size)
            direct_background = t0_rgb.copy()
            direct_background[background] = layers["first_rgb"][background]
            Image.fromarray(np.uint8(np.clip(direct_background * 255, 0, 255))).save(
                out / "t0_background_layer_direct_composite.png"
            )
            Image.fromarray(np.uint8(np.clip(layers["first_rgb"] * 255, 0, 255))).save(out / "da3_foreground_rgb.png")
            Image.fromarray(np.uint8(np.clip(layers["second_rgb"] * 255, 0, 255))).save(out / "da3_background_rgb.png")
            Image.fromarray(np.uint8(np.clip(layers["first_rgb"] * 255, 0, 255))).save(out / "background_rgb_nearest.png")
            Image.fromarray(np.uint8(np.clip(layers["second_rgb"] * 255, 0, 255))).save(out / "background_rgb_second.png")
            Image.fromarray(_depth_vis(layers["first_depth"])).save(out / "background_depth_nearest.png")
            Image.fromarray(_depth_vis(layers["second_depth"])).save(out / "background_depth_second.png")
            np.save(out / "da3_foreground_surface_depth.npy", layers["first_depth"])
            np.save(out / "da3_background_surface_depth.npy", layers["second_depth"])
            np.save(out / "da3_foreground_semantic.npy", layers["first_semantic"].astype(np.float16))
            np.save(out / "da3_background_semantic.npy", layers["second_semantic"].astype(np.float16))
            Image.fromarray(np.uint8(_semantic_vis(layers["first_semantic"]) * 255)).save(out / "da3_foreground_semantic_pca.png")
            Image.fromarray(np.uint8(_semantic_vis(layers["second_semantic"]) * 255)).save(out / "da3_background_semantic_pca.png")
            Image.fromarray(np.uint8(background * 255)).save(out / "background_supportable_mask.png")
            Image.fromarray(np.uint8(triple_visibility * 255)).save(out / "triple_reprojection_visibility.png")
            Image.fromarray(np.uint8(behind * 255)).save(out / "distinct_second_layer_mask.png")
            Image.fromarray(np.uint8(true_disocclusion * 255)).save(out / "true_disocclusion_mask.png")
            _save_sheet(out / "background_layer_diagnostic.jpg", [
                ("T0 residual hole", hole.astype(np.float32)),
                ("nearest background RGB", layers["first_rgb"]),
                ("second background RGB", layers["second_rgb"]),
                ("nearest background depth", _depth_vis(layers["first_depth"])),
                ("second background depth", _depth_vis(layers["second_depth"])),
                ("DA3 foreground semantic", _semantic_vis(layers["first_semantic"])),
                ("DA3 background semantic", _semantic_vis(layers["second_semantic"])),
                ("supportable background", background.astype(np.float32)),
                ("triple visibility", triple_visibility.astype(np.float32)),
                ("true disocclusion", true_disocclusion.astype(np.float32)),
            ])
            report[pose][target_name] = {
                "resolution": list(size),
                "residual_hole_fraction": float(hole.mean()),
                "background_supportable_fraction_of_hole": float(background.sum() / max(hole.sum(), 1)),
                "distinct_second_layer_fraction_of_hole": float(behind.sum() / max(hole.sum(), 1)),
                "true_disocclusion_fraction_of_hole": float(true_disocclusion.sum() / max(hole.sum(), 1)),
                "triple_visibility_fraction_of_hole": float((triple_visibility & hole).sum() / max(hole.sum(), 1)),
                "source_topology": list(source_names),
                "contract": f"DA3-first diagnostic ({args.depth_layer}); triple_reprojection={args.triple_reprojection}; no T0 pixels modified; RGB/semantic sampled from source surfaces",
            }
            print(json.dumps({"pose": pose, "camera": target_name, **report[pose][target_name]}, ensure_ascii=False), flush=True)
    (args.output / "report.json").write_text(json.dumps({
        "sequence": sequence, "frame": frame, "scale": args.scale, "margin_m": args.margin_m,
        "depth_layer": args.depth_layer,
        "triple_reprojection": args.triple_reprojection,
        "triple_pixel_tolerance": args.triple_pixel_tolerance,
        "semantic_contract": "dino_anyup_semantic_320x180.npy is used as feature tensor; PCA PNG is visualization only",
        "report": report,
    }, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
