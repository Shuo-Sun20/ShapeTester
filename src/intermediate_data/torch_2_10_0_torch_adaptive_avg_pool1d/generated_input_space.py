import torch
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import List

def call_func(inputs, output_size):
    return F.adaptive_avg_pool1d(inputs, output_size)

example_input = torch.randn(2, 3, 10)
example_output = call_func(example_input, 5)

valid_test_case = {
    "inputs": torch.randn(2, 3, 10),
    "output_size": 5
}

@dataclass
class InputSpace:
    output_size: List[int] = field(default_factory=lambda: [1, 3, 5, 7, 10])