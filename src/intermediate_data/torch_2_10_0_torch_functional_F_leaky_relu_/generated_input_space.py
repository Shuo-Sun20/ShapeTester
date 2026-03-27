import torch
from dataclasses import dataclass

valid_test_case = {"inputs": torch.randn(3, 4), "negative_slope": 0.1}

@dataclass
class InputSpace:
    pass