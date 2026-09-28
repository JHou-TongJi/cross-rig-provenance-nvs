"""Evaluate plane-sweep and Foundation V2 ablations on identical samples."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader

from gcr_nvs.datasets.torch_dataset import LeaveOneOutTorchDataset
from gcr_nvs.evaluation.metrics import psnr, ssim_proxy
from gcr_nvs.models.foundation_v2 import FoundationV2SingleFrame
from gcr_nvs.rendering.plane_sweep import PlaneSweepRenderer


def _calibrations(batch, device):
    source_K = batch['source_intrinsics'].to(device)
    source_T = batch['source_extrinsics'].to(device)
    source_D = batch['source_distortions'].to(device)
    source = [
        {'K': source_K[:, index], 'D': source_D[:, index], 'T_world': source_T[:, index]}
        for index in range(source_K.shape[1])
    ]
    target = {
        'K': batch['target_intrinsic'].to(device),
        'D': batch['target_distortion'].to(device),
        'T_world': batch['target_extrinsic'].to(device),
    }
    return source, target


def _save_rgb(tensor: torch.Tensor, path: Path):
    image = tensor.detach().cpu().clamp(0, 1)[0].permute(1, 2, 0).numpy()
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((image * 255).astype('uint8')).save(path)


def _save_gray(tensor: torch.Tensor, path: Path):
    image = tensor.detach().cpu().clamp(0, 1)[0, 0].numpy()
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((image * 255).astype('uint8')).save(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('root', type=Path)
    parser.add_argument('checkpoint', type=Path)
    parser.add_argument('--model', choices=('plane_sweep', 'foundation_v2'), required=True)
    parser.add_argument('--sequences', nargs='+', required=True)
    parser.add_argument('--distortion', type=Path)
    parser.add_argument('--target-rig', type=Path)
    parser.add_argument('--target-camera', default='CAM_FRONT_WIDE')
    parser.add_argument('--all-target-cameras', action='store_true')
    parser.add_argument(
        '--exclude-nearest-source',
        action='store_true',
        help='Remove the pose-nearest source after leave-one-out selection.',
    )
    parser.add_argument('--width', type=int, default=256)
    parser.add_argument('--height', type=int, default=128)
    parser.add_argument('--depth-layers', type=int, default=16)
    parser.add_argument('--max-points', type=int, default=4096)
    parser.add_argument('--max-samples', type=int)
    parser.add_argument('--samples-per-sequence', type=int)
    parser.add_argument('--num-workers', type=int, default=2)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--images', type=Path)
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    target_camera = None if args.all_target_cameras else args.target_camera
    dataset = LeaveOneOutTorchDataset(
        args.root, args.sequences, args.distortion, args.width, args.height,
        max_points=args.max_points, max_samples=args.max_samples,
        target_rig=args.target_rig, target_camera=target_camera,
        foundation_minimal_inputs=True,
        balance_target_cameras=args.all_target_cameras,
        exclude_nearest_source=args.exclude_nearest_source,
        max_samples_per_sequence=args.samples_per_sequence,
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=args.num_workers, pin_memory=device == 'cuda')
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    state = checkpoint.get('model', checkpoint)
    if args.model == 'plane_sweep':
        model = PlaneSweepRenderer(num_depth_layers=args.depth_layers).to(device)
    else:
        model = FoundationV2SingleFrame(plane_sweep_depth_layers=args.depth_layers).to(device)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if unexpected:
        raise RuntimeError(f'unexpected checkpoint keys: {unexpected}')
    if missing:
        print(json.dumps({'missing_checkpoint_keys': missing}), flush=True)
    model.eval()

    rows = []
    with torch.no_grad():
        for index, batch in enumerate(loader):
            source_images = batch['source_images'].to(device)
            target_rgb = batch['target_rgb'].to(device)
            geometry = batch['geometry'].to(device)
            camera = batch['target_camera'][0]
            source_cameras = batch['source_cameras'][0].split('|')
            source_calibs, target_calib = _calibrations(batch, device)
            if args.model == 'plane_sweep':
                output = model(source_images, source_calibs, target_calib, geometry[:, 3:4])
                coarse = final = output['rgb']
                coverage = output['valid_mask']
            else:
                output = model(source_images, source_calibs, target_calib, batch['ray_map'].to(device), geometry[:, 3:4], geometry)
                coarse, final = output['coarse_rgb'], output['rgb']
                coverage = output['target_geometry'][:, 7:8]
            rows.append({
                'camera': camera,
                'source_cameras': source_cameras,
                'coarse_psnr': psnr(coarse, target_rgb),
                'coarse_ssim': ssim_proxy(coarse, target_rgb),
                'final_psnr': psnr(final, target_rgb),
                'final_ssim': ssim_proxy(final, target_rgb),
                'coverage': float(coverage.mean()),
                'observed_psnr': psnr(final, target_rgb, coverage),
                'hole_psnr': psnr(final, target_rgb, 1.0 - coverage),
            })
            if args.images is not None and index == 0:
                _save_rgb(target_rgb, args.images / 'target_rgb.png')
                _save_rgb(coarse, args.images / 'coarse_rgb.png')
                _save_rgb(final, args.images / 'final_rgb.png')
                if 'seam_mask' in output:
                    _save_gray(output['seam_mask'], args.images / 'seam_mask.png')

    report = {
        'model': args.model,
        'evaluation_task': 'strict leave-one-camera-out reconstruction',
        'exclude_nearest_source': args.exclude_nearest_source,
        'samples': len(rows),
        **{
            key: sum(row[key] for row in rows) / max(1, len(rows))
            for key in rows[0]
            if key not in {'camera', 'source_cameras'}
        },
        'per_camera': {
            camera: {
                key: sum(row[key] for row in rows if row['camera'] == camera)
                / max(sum(row['camera'] == camera for row in rows), 1)
                for key in rows[0]
                if key not in {'camera', 'source_cameras'}
            }
            for camera in sorted({row['camera'] for row in rows})
        },
        'per_sample': rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({key: value for key, value in report.items() if key != 'per_sample'}, indent=2))


if __name__ == '__main__':
    main()
