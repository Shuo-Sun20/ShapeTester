import torch
import torch.nn as nn
from dataclasses import dataclass, field
from typing import Union, List, Tuple

def call_func(padding, inputs):
    pad_layer = nn.ReflectionPad1d(padding)
    return pad_layer(inputs)

# Valid test case from the example
torch.manual_seed(42)
example_input = torch.randn(1, 2, 4)
valid_test_case = {
    "padding": 2,
    "inputs": example_input
}

@dataclass
class InputSpace:
    """Dataclass containing all parameters affecting output tensor shape with discretized value ranges."""
    # Padding can be either int or tuple (padding_left, padding_right)
    # The constraint: padding size must be less than corresponding input dimension
    padding: List[Union[int, Tuple[int, int]]] = field(
        default_factory=lambda: [
            # Discrete values - directly list all possible patterns
            0,  # No padding
            1,  # Minimal positive symmetric padding
            2,  # Example value from valid test case
            3,  # Medium symmetric padding
            4,  # Larger symmetric padding
            
            # Tuple patterns covering different scenarios
            (0, 1),  # Asymmetric - pad only right
            (1, 0),  # Asymmetric - pad only left
            (1, 2),  # Asymmetric - different paddings
            (2, 1),  # Asymmetric - reversed different paddings
            (3, 1),  # Example from documentation
            (0, 0),  # Zero padding (explicit tuple form)
            
            # Boundary cases (assuming typical input dimensions of 4-16)
            (1, 1),  # Minimal symmetric tuple padding
            (3, 3),  # Maximum typical symmetric padding for small inputs
        ]
    )