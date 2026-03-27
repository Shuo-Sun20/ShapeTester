import torch
from dataclasses import dataclass

valid_test_case = {
    'inputs': torch.tensor([-1 + 1j, -2 + 2j, 3 - 3j], dtype=torch.complex64)
}

@dataclass
class InputSpace:
    pass