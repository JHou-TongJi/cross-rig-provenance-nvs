"""Inference-only RGB-D-S completion helpers.

The training implementation may learn the adapter, but deployment only needs
these frozen-pipeline functions. Keeping them here lets the inference-only
release remove all optimizer/loss code without breaking hole completion.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F
from diffusers import DDIMScheduler, T2IAdapter


PROMPT = "a realistic undistorted automotive camera road scene, natural details"


def load_adapter(path: Path, device: torch.device) -> T2IAdapter:
    model = T2IAdapter(
        in_channels=16, channels=[320, 640, 1280, 1280], num_res_blocks=2,
        downscale_factor=8, adapter_type="full_adapter",
    ).to(device=device, dtype=torch.float32).eval()
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=False)["adapter"])
    return model


def prompt_embeddings(pipe, device: torch.device) -> torch.Tensor:
    tokens = pipe.tokenizer([PROMPT], padding="max_length", max_length=pipe.tokenizer.model_max_length,
                            truncation=True, return_tensors="pt").input_ids.to(device)
    with torch.inference_mode():
        encoded = pipe.text_encoder(tokens)[0]
    return encoded.clone()


def _vae_latents(pipe, rgb: torch.Tensor) -> torch.Tensor:
    with torch.inference_mode():
        latent = pipe.vae.encode(rgb * 2.0 - 1.0).latent_dist.sample()
    return (latent * pipe.vae.config.scaling_factor).clone()


@torch.inference_mode()
def sample(pipe, adapter, batch, prompt, variant: str, steps: int, seed: int):
    device = prompt.device; dtype = pipe.unet.dtype
    masked = batch["masked_rgb"].to(device=device, dtype=dtype)
    hole = batch["hole_mask"].to(device=device, dtype=dtype)
    control = batch["control"].to(device=device, dtype=dtype)
    if variant == "a":
        control = control.zero_()
    elif variant == "d":
        pass
    else:
        raise ValueError(f"inference route supports a or d, got {variant}")
    masked_latent = _vae_latents(pipe, masked).to(dtype)
    latent_mask = F.interpolate(hole, size=masked_latent.shape[-2:], mode="nearest")
    scheduler = DDIMScheduler.from_config(pipe.scheduler.config); scheduler.set_timesteps(steps, device=device)
    generator = torch.Generator(device=device).manual_seed(seed)
    latent = torch.randn(masked_latent.shape, generator=generator, device=device, dtype=dtype) * scheduler.init_noise_sigma
    with torch.autocast(device_type="cuda", dtype=dtype):
        residuals = adapter(control)
    for timestep in scheduler.timesteps:
        model_input = torch.cat([scheduler.scale_model_input(latent, timestep), latent_mask, masked_latent], dim=1)
        with torch.autocast(device_type="cuda", dtype=dtype):
            noise = pipe.unet(model_input, timestep, prompt,
                              down_intrablock_additional_residuals=[item for item in residuals]).sample
        latent = scheduler.step(noise, timestep, latent).prev_sample
    decoded = pipe.vae.decode(latent / pipe.vae.config.scaling_factor).sample
    proposal = (decoded * 0.5 + 0.5).clamp(0, 1)
    final = masked * (1.0 - hole) + proposal * hole
    return masked, hole, proposal, final
