import torch
from dataclasses import dataclass, field
from typing import Union, List, Tuple

def call_func(padding, value, inputs):
    if isinstance(inputs, list):
        input_tensor = inputs[0]
    else:
        input_tensor = inputs
    
    pad_layer = torch.nn.ConstantPad3d(padding, value)
    output = pad_layer(input_tensor)
    return output

# 1. Define valid_test_case dict
valid_test_case = {
    'padding': (3, 3, 6, 6, 0, 1),
    'value': 3.5,
    'inputs': torch.randn(16, 3, 10, 20, 30)
}

# 2. Parameters affecting output shape (except inputs): padding

# 3. Discretized value space for padding
@dataclass
class InputSpace:
    # Padding can be int or tuple of 6 ints
    # Discrete parameter with possible values including:
    # - Single integer (same padding on all 6 sides)
    # - 6-element tuple for different padding per side
    padding: List[Union[int, Tuple[int, int, int, int, int, int]]] = field(
        default_factory=lambda: [
            # Single integer cases (symmetric padding)
            0,  # No padding
            1,  # Minimal padding
            2,  # Small padding
            3,  # Example from valid_test_case (partial)
            5,  # Medium padding
            10, # Large padding
            
            # Tuple cases (asymmetric padding)
            (0, 0, 0, 0, 0, 0),  # No padding
            (1, 1, 1, 1, 1, 1),  # Uniform padding
            (3, 3, 6, 6, 0, 1),  # Example from valid_test_case
            (1, 2, 3, 4, 5, 6),  # All different values
            (10, 5, 0, 0, 2, 2), # Mixed zero and non-zero
            (0, 0, 0, 0, 5, 0),  # Single side padding
        ]
    )