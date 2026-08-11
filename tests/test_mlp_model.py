"""Basic neural network shape / forward tests."""
import torch
import torch.nn as nn


class SupplyChainMLP(nn.Module):
    """Minimal copy of the project MLP for isolated unit testing."""

    def __init__(self, input_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze()


def test_mlp_forward_shape_batch():
    model = SupplyChainMLP(input_dim=20)
    model.eval()
    x = torch.randn(8, 20)
    with torch.no_grad():
        y = model(x)
    assert y.shape == (8,)


def test_mlp_forward_shape_single():
    model = SupplyChainMLP(input_dim=10)
    model.eval()
    x = torch.randn(1, 10)
    with torch.no_grad():
        y = model(x)
    # squeeze may yield 0-dim tensor for batch size 1
    assert y.numel() == 1


def test_mlp_parameters_are_trainable():
    model = SupplyChainMLP(input_dim=5)
    params = list(model.parameters())
    assert len(params) > 0
    assert all(p.requires_grad for p in params)
