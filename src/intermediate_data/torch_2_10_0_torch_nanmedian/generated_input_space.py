import torch

example_tensor = torch.tensor([[2.0, 3.0, float('nan'), 1.0],
                               [float('nan'), 5.0, 4.0, float('nan')]])

valid_test_case = {
    'inputs': example_tensor,
    'dim': 1,
    'keepdim': False,
    'out': None
}

from dataclasses import dataclass, field

@dataclass
class InputSpace:
    dim: list = field(default_factory=lambda: [None, 0, 1, -1, -2])
    keepdim: list = field(default_factory=lambda: [True, False])