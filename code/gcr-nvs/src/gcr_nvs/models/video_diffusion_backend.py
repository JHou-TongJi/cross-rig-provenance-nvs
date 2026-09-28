"""Optional video-diffusion backend contract for hole-only completion.

The deterministic T0/DA3 route remains the default.  The Wan/VACE backend is
intentionally lazy: importing this module does not import diffusers or allocate
GPU memory.  A future adapter can map the condition bundle to the UniWorld-
View pipeline without changing the T0 observed-pixel lock.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol

import torch


@dataclass(frozen=True)
class VideoDiffusionConfig:
    backend: str = "disabled"
    enabled: bool = False
    model_id: str | None = None
    uniview_repo: Path | None = None
    device: str = "cuda"
    dtype: str = "float16"
    max_frames: int = 16
    inference_steps: int = 8
    cpu_offload: bool = True
    local_files_only: bool = True
    hole_only: bool = True
    observed_lock: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any] | None) -> "VideoDiffusionConfig":
        values = dict(mapping or {})
        repo = values.get("uniview_repo")
        if repo is not None:
            repo = Path(repo)
        return cls(
            backend=str(values.get("backend", "disabled")),
            enabled=bool(values.get("enabled", False)),
            model_id=values.get("model_id"),
            uniview_repo=repo,
            device=str(values.get("device", "cuda")),
            dtype=str(values.get("dtype", "float16")),
            max_frames=int(values.get("max_frames", 16)),
            inference_steps=int(values.get("inference_steps", 8)),
            cpu_offload=bool(values.get("cpu_offload", True)),
            local_files_only=bool(values.get("local_files_only", True)),
            hole_only=bool(values.get("hole_only", True)),
            observed_lock=bool(values.get("observed_lock", True)),
            metadata=dict(values.get("metadata", {})),
        )

    def validate(self) -> None:
        if self.backend not in {"disabled", "wan_vace"}:
            raise ValueError(f"unsupported video diffusion backend: {self.backend}")
        if self.backend == "wan_vace" and self.enabled and not self.model_id:
            raise ValueError("wan_vace requires model_id when enabled")
        if self.max_frames < 1:
            raise ValueError("max_frames must be positive")
        if self.inference_steps < 1:
            raise ValueError("inference_steps must be positive")
        if not self.hole_only or not self.observed_lock:
            raise ValueError("video diffusion must be hole_only with observed_lock enabled")


class VideoCompletionBackend(Protocol):
    name: str

    def load(self) -> None:
        ...

    def complete(
        self,
        conditions: Mapping[str, torch.Tensor],
        *,
        hole_mask: torch.Tensor,
        seed: int | None = None,
    ) -> torch.Tensor:
        ...


class DisabledVideoCompletionBackend:
    name = "disabled"

    def load(self) -> None:
        return None

    def complete(
        self,
        conditions: Mapping[str, torch.Tensor],
        *,
        hole_mask: torch.Tensor,
        seed: int | None = None,
    ) -> torch.Tensor:
        """Return the deterministic RGB condition without changing observed pixels."""
        del seed
        if "rgb" not in conditions:
            raise KeyError("disabled backend requires conditions['rgb']")
        rgb = conditions["rgb"]
        expected_mask_shape = tuple(rgb.shape[:1]) + (1,) + tuple(rgb.shape[2:])
        if rgb.ndim != 4 or tuple(hole_mask.shape) != expected_mask_shape:
            raise ValueError("rgb must be BCHW and hole_mask must be B1HW")
        return rgb


class LazyWanVACEBackend:
    """Lazy integration point for UniWorld-View's Wan/VACE pipeline.

    Loading is explicit and opt-in.  The backend accepts a ``pipeline_factory``
    so a future project-specific condition adapter can be injected without
    coupling the deterministic T0 code to the external repository API.
    """

    name = "wan_vace"

    def __init__(
        self,
        config: VideoDiffusionConfig,
        pipeline_factory: Callable[[VideoDiffusionConfig], Any] | None = None,
    ) -> None:
        config.validate()
        self.config = config
        self.pipeline_factory = pipeline_factory
        self.pipeline: Any | None = None
        self.loaded = False

    def load(self) -> None:
        if not self.config.enabled:
            raise RuntimeError("wan_vace backend is configured but disabled")
        if self.pipeline_factory is not None:
            self.pipeline = self.pipeline_factory(self.config)
            self.loaded = True
            return
        if self.config.uniview_repo is None:
            raise RuntimeError("set uniview_repo or provide pipeline_factory before loading Wan/VACE")
        repo = self.config.uniview_repo.expanduser().resolve()
        if not repo.exists():
            raise FileNotFoundError(f"UniWorld-View repository not found: {repo}")
        try:
            import sys
            if str(repo) not in sys.path:
                sys.path.insert(0, str(repo))
            from model.pipeline_uniview import WanVACEPipeline
        except ImportError as exc:
            raise RuntimeError(
                "UniWorld-View dependencies are unavailable; install its optional environment "
                "or pass a project-specific pipeline_factory"
            ) from exc
        if not self.config.model_id:
            raise ValueError("model_id is required to load Wan/VACE")
        dtype = getattr(torch, self.config.dtype, torch.float16)
        self.pipeline = WanVACEPipeline.from_pretrained(
            self.config.model_id,
            torch_dtype=dtype,
            local_files_only=self.config.local_files_only,
        )
        if self.config.cpu_offload and hasattr(self.pipeline, "enable_model_cpu_offload"):
            self.pipeline.enable_model_cpu_offload()
        elif hasattr(self.pipeline, "to"):
            self.pipeline.to(self.config.device)
        self.loaded = True

    def complete(
        self,
        conditions: Mapping[str, torch.Tensor],
        *,
        hole_mask: torch.Tensor,
        seed: int | None = None,
    ) -> torch.Tensor:
        if not self.loaded or self.pipeline is None:
            raise RuntimeError("Wan/VACE backend is not loaded; call load() explicitly")
        if not self.config.hole_only or not self.config.observed_lock:
            raise RuntimeError("refusing a Wan/VACE call without hole-only and observed-lock contracts")
        if "rgb" not in conditions:
            raise KeyError("Wan/VACE conditions require an rgb tensor")
        raise NotImplementedError(
            "Provide a project-specific condition adapter that maps "
            "rgb/depth/visibility/semantic/reference tokens to WanVACEPipeline. "
            "The backend deliberately refuses an unsafe generic full-frame call."
        )


def build_video_completion_backend(
    config: VideoDiffusionConfig | Mapping[str, Any] | None = None,
    *,
    pipeline_factory: Callable[[VideoDiffusionConfig], Any] | None = None,
) -> VideoCompletionBackend:
    resolved = config if isinstance(config, VideoDiffusionConfig) else VideoDiffusionConfig.from_mapping(config)
    resolved.validate()
    if resolved.backend == "disabled" or not resolved.enabled:
        return DisabledVideoCompletionBackend()
    if resolved.backend == "wan_vace":
        return LazyWanVACEBackend(resolved, pipeline_factory=pipeline_factory)
    raise ValueError(f"unsupported video diffusion backend: {resolved.backend}")
