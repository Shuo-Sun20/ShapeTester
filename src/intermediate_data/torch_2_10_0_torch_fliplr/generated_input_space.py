import torch
from dataclasses import dataclass

valid_test_case = {"inputs": torch.randn(3, 4)}

@dataclass
class InputSpace:
    pass