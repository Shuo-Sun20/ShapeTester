import torch
from dataclasses import dataclass, field

valid_test_case = {
    "inputs": [torch.tensor([3.4742, 0.5466, -0.8008, -0.9079])],
    "out": None
}

@dataclass
class InputSpace:
    out: list = field(default_factory=lambda: [
        None,
        torch.tensor([0., 0., 0., 0.]),
        torch.tensor([1., 1., 1., 1.]),
        torch.tensor([-1., -1., -1., -1.]),
        torch.tensor([float('inf'), float('inf'), float('inf'), float('inf')]),
        torch.tensor([float('-inf'), float('-inf'), float('-inf'), float('-inf')]),
        torch.full((4,), float('nan')),
        torch.tensor([3., 0., -0., -0.], dtype=torch.float64),
        torch.tensor([3., 0., -0., -0.], dtype=torch.float32),
        torch.tensor([3., 0., -0., -0.], dtype=torch.float16)
    ])