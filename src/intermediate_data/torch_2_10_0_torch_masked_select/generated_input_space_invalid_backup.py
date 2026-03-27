from dataclasses import dataclass, field
import torch

# Task 1: Define valid_test_case
input_tensor = torch.tensor([[ 0.3552, -2.3825, -0.8297,  0.3477],
                             [-1.2035,  1.2252,  0.5002,  0.6248],
                             [ 0.1307, -2.0608,  0.1244,  2.0139]])
mask_tensor = input_tensor.ge(0.5)
valid_test_case = {
    "inputs": [input_tensor, mask_tensor],
    "out": None
}

# Task 2-4: Define InputSpace dataclass
@dataclass
class InputSpace:
    out: list = field(default_factory=lambda: [
        None,
        torch.tensor([1.2252, 0.5002, 0.6248, 2.0139], dtype=torch.float32),
        torch.tensor([0., 0., 0., 0.], dtype=torch.float32),
        torch.tensor([-1., -1., -1., -1.], dtype=torch.float32),
        torch.empty(0, dtype=torch.float32)
    ])