import torch
from dataclasses import dataclass, field
from typing import Any, List, Union

# Task 1: Define valid_test_case
torch.manual_seed(42)
valid_test_case = {
    'constructor_args': [54],
    'constructor_kwargs': {'unused_argument1': 0.1, 'unused_argument2': False},
    'inputs': torch.randn(128, 20)
}

# Task 2-4: Define InputSpace dataclass
@dataclass
class InputSpace:
    """
    Contains all parameters that affect the shape of the output tensor.
    For torch.nn.Identity, no parameters affect output shape except inputs,
    but we include constructor_args for completeness.
    """
    # constructor_args can be any list of arguments (all ignored by Identity)
    # We discretize to show different argument types and counts
    constructor_args: List[Any] = field(
        default_factory=lambda: [
            [],                     # No arguments
            [54],                   # Single integer
            [3.14],                 # Single float
            [10, 20],               # Multiple integers
            ['test'],               # String (ignored)
            [torch.tensor([1, 2])] # Tensor argument (ignored)
        ]
    )
    
    # constructor_kwargs can be any dict (all ignored by Identity)
    # We provide different keyword argument configurations
    constructor_kwargs: List[dict] = field(
        default_factory=lambda: [
            {},                                      # No kwargs
            {'unused_arg1': 0.1},                    # Single kwarg
            {'arg1': 1, 'arg2': False},             # Multiple kwargs
            {'arg': None},                          # None value
            {'arg': [1, 2, 3]},                     # List value
            {'unused_argument1': 0.1, 'unused_argument2': False}  # Example case
        ]
    )

# The class can be successfully instantiated
var = InputSpace()