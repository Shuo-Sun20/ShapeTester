import torch
from dataclasses import dataclass, field
from typing import Tuple

def call_func(inputs, dim, sizes):
    return torch.unflatten(inputs, dim, sizes)

x = torch.randn(3, 4, 1)
example_output = call_func(x, 1, (2, 2))

# 1. Define valid_test_case
valid_test_case = {
    "inputs": torch.randn(3, 4, 1),
    "dim": 1,
    "sizes": (2, 2)
}

# 2. Parameters affecting output shape (except inputs): dim and sizes

# 3. Discretized value spaces for dim and sizes
# dim: integer, can be negative or positive, must be within valid index range
# For a 3D tensor example (shape [3,4,1]), valid dims: -3, -2, -1, 0, 1, 2
# We'll generalize to include boundary values and typical cases
dim_values = [-3, -2, -1, 0, 1, 2, 5, -5]  # Includes invalid cases (5, -5) for completeness

# sizes: tuple of ints, product must equal input.shape[dim], can contain -1
# For dim=1 with input size 4, possible size tuples:
sizes_values = [
    (2, 2),    # valid, product = 4
    (4, 1),    # valid, product = 4
    (1, 4),    # valid, product = 4
    (-1, 2),   # valid, infers 2
    (2, -1),   # valid, infers 2
    (-1, 4),   # valid, infers 1
    (4, -1),   # valid, infers 1
    (1, 2, 2), # valid, product = 4
    (2, 1, 2), # valid, product = 4
    (2, 2, 1), # valid, product = 4
    (1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1),  # 16 ones, product = 1 (invalid for dim=1, size=4)
    (1,),      # invalid for dim=1, size=4 (product=1)
    (5, 1),    # invalid for dim=1, size=4 (product=5)
]

# 4. Define InputSpace dataclass
@dataclass
class InputSpace:
    dim: list = field(default_factory=lambda: [-3, -2, -1, 0, 1, 2, 5, -5])
    sizes: list = field(default_factory=lambda: [
        (2, 2),
        (4, 1),
        (1, 4),
        (-1, 2),
        (2, -1),
        (-1, 4),
        (4, -1),
        (1, 2, 2),
        (2, 1, 2),
        (2, 2, 1),
        (1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1),
        (1,),
        (5, 1),
    ])