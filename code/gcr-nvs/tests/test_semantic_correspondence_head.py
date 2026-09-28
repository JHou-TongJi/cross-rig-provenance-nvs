import torch

from gcr_nvs.models.semantic_correspondence_head import SemanticCorrespondenceHead


def test_correspondence_head_shape_and_gradients() -> None:
    model = SemanticCorrespondenceHead(input_dim=7, hidden_dim=12)
    features = torch.randn(2, 5, 7, requires_grad=True)
    logits = model(features)
    assert logits.shape == (2, 5)
    logits.mean().backward()
    assert features.grad is not None
