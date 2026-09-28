import torch
import pytest

from gcr_nvs.models.video_diffusion_backend import (
    DisabledVideoCompletionBackend,
    LazyWanVACEBackend,
    VideoDiffusionConfig,
    build_video_completion_backend,
)


def test_default_backend_is_disabled_and_identity():
    backend = build_video_completion_backend()
    assert isinstance(backend, DisabledVideoCompletionBackend)
    rgb = torch.rand(2, 3, 8, 8)
    output = backend.complete({"rgb": rgb}, hole_mask=torch.ones(2, 1, 8, 8))
    assert torch.equal(output, rgb)


def test_wan_backend_is_lazy_and_does_not_load_until_explicit_call():
    backend = build_video_completion_backend({
        "backend": "wan_vace",
        "enabled": True,
        "model_id": "local/test-model",
        "uniview_repo": "/tmp/does-not-need-to-exist-yet",
    })
    assert isinstance(backend, LazyWanVACEBackend)
    assert backend.loaded is False


def test_wan_backend_rejects_unsafe_contract():
    with pytest.raises(ValueError):
        VideoDiffusionConfig.from_mapping({
            "backend": "wan_vace",
            "enabled": True,
            "model_id": "x",
            "observed_lock": False,
        }).validate()


def test_wan_backend_requires_explicit_load():
    backend = build_video_completion_backend({
        "backend": "wan_vace",
        "enabled": True,
        "model_id": "local/test-model",
        "uniview_repo": "/tmp/does-not-need-to-exist-yet",
    })
    with pytest.raises(RuntimeError, match="not loaded"):
        backend.complete({"rgb": torch.rand(1, 3, 4, 4)}, hole_mask=torch.ones(1, 1, 4, 4))
