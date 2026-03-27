from dataclasses import dataclass, field
import torch

# Task 1: Define valid_test_case
elements = torch.randint(0, 10, (3, 4))
test_elements = torch.tensor([2, 5, 8])
valid_test_case = {
    'inputs': [elements, test_elements],
    'assume_unique': False,
    'invert': False
}

# Task 2 & 3: Parameters affecting output shape and their value spaces
# Only 'assume_unique' and 'invert' can affect output shape (through their boolean values)

# Task 4: Define InputSpace dataclass
@dataclass
class InputSpace:
    assume_unique: list = field(default_factory=lambda: [True, False])
    invert: list = field(default_factory=lambda: [True, False])