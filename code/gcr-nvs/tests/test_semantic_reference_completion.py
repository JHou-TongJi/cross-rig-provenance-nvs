import torch

from gcr_nvs.models.deepfill_v2_reference import (
    DeepFillPatchDiscriminator,
    SemanticDeepFillV2,
    discriminator_hinge_loss,
    generator_hinge_loss,
)
from gcr_nvs.models.semantic_reference_completion import (
    LocalSemanticReferenceRetriever,
    strict_mask_composite,
)


def test_strict_mask_composite_preserves_observed_pixels_exactly() -> None:
    observed = torch.rand(1, 3, 12, 16)
    proposal = torch.rand_like(observed)
    mask = torch.zeros(1, 1, 12, 16, dtype=torch.bool)
    mask[..., 3:8, 5:11] = True
    result = strict_mask_composite(observed, proposal, mask)
    assert torch.equal(result[~mask.expand_as(result)], observed[~mask.expand_as(observed)])
    assert torch.equal(result[mask.expand_as(result)], proposal[mask.expand_as(proposal)])


def test_semantic_retriever_copies_matching_reference_rgb() -> None:
    query = torch.zeros(1, 2, 8, 10)
    query[:, 0] = 1.0
    references = torch.zeros(1, 2, 2, 8, 10)
    references[:, 0, 1] = 1.0
    references[:, 1, 0] = 1.0
    rgb = torch.zeros(1, 2, 3, 8, 10)
    rgb[:, 1, 2] = 0.75
    hole = torch.ones(1, 1, 8, 10, dtype=torch.bool)
    retriever = LocalSemanticReferenceRetriever(radius=0, minimum_confidence=0.0)
    result = retriever(query, references, rgb, hole)
    assert torch.all(result["source_index"] == 1)
    assert torch.allclose(result["retrieved_rgb"][:, 2], torch.full((1, 8, 10), 0.75))
    assert result["recoverable_mask"].all()


def test_deepfill_v2_reference_output_contract() -> None:
    model = SemanticDeepFillV2(
        semantic_channels=4, geometry_channels=3, base_channels=8, attention_radius=1,
    ).eval()
    t0 = torch.rand(1, 3, 31, 47)
    hole = torch.zeros(1, 1, 31, 47)
    hole[..., 8:22, 13:34] = 1.0
    with torch.no_grad():
        output = model(
            t0,
            hole,
            torch.rand_like(t0),
            torch.rand(1, 1, 31, 47),
            torch.rand(1, 4, 31, 47),
            torch.rand(1, 3, 31, 47),
        )
    assert output["rgb"].shape == t0.shape
    observed = ~hole.bool().expand_as(t0)
    assert torch.equal(output["rgb"][observed], t0[observed])
    assert torch.isfinite(output["rgb"]).all()


def test_deepfill_discriminator_and_hinge_losses() -> None:
    discriminator = DeepFillPatchDiscriminator(base_channels=8).eval()
    rgb = torch.rand(1, 3, 64, 64)
    mask = torch.zeros(1, 1, 64, 64)
    logits = discriminator(rgb, mask)
    assert logits.ndim == 4 and logits.shape[:2] == (1, 1)
    assert torch.isfinite(discriminator_hinge_loss(logits, logits)).all()
    assert torch.isfinite(generator_hinge_loss(logits)).all()
