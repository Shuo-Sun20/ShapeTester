import torch
from dataclasses import dataclass, field

torch.manual_seed(42)
example_input = torch.randn(16, dtype=torch.complex64)

valid_test_case = {
    "inputs": example_input,
    "n": None,
    "dim": -1,
    "norm": None,
    "out": None
}

@dataclass
class InputSpace:
    # Parameters that affect output shape (excluding 'inputs'):
    n: list = field(default_factory=lambda: [None, 1, 4, 8, 16, 32, 64, 128])  # Signal length (powers of 2)
    dim: list = field(default_factory=lambda: [0, -1, -2, 1, 2])  # Dimension indices (including negative)