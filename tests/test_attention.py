import torch

from src.circuitLens.analysis.metrics import AttentionMetrics


def test_attention_entropy():
    attention = torch.tensor([[[0.5, 0.5]]])

    result = AttentionMetrics.attention_entropy(attention)

    assert result.shape == torch.Size([1, 1])
    assert result.item() > 0


def test_maximum_attention():
    attention = torch.tensor([[[0.2, 0.8]]])

    result = AttentionMetrics.maximum_attention(attention)

    assert abs(result.item() -0.8)<1e-6