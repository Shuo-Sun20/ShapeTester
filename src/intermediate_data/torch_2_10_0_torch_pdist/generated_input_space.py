import torch
from dataclasses import dataclass
from typing import List

def call_func(inputs, p=2):
    return torch.pdist(inputs, p=p)

# Task 1: Define a valid test case
valid_test_case = {
    "inputs": torch.randn(5, 3),
    "p": 2
}

# Task 4: Define the InputSpace dataclass
@dataclass
class InputSpace:
    pass