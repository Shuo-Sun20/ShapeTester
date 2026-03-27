import torch
from dataclasses import dataclass, field
from typing import Union, Tuple, List

# Task 1: Define valid_test_case
torch.manual_seed(0)
input_tensor = torch.randn(1, 1, 4, 4)
pool = torch.nn.MaxPool2d(2, stride=2, return_indices=True)
pooled_output, indices = pool(input_tensor)

valid_test_case = {
    'kernel_size': 2,
    'stride': 2,
    'padding': 0,
    'inputs': [pooled_output, indices]
}

# Task 4: Define InputSpace dataclass
@dataclass
class InputSpace:
    """Dataclass containing discretized value ranges for shape-affecting parameters."""
    
    # kernel_size: int or tuple of ints (height, width)
    kernel_size: List[Union[int, Tuple[int, int]]] = field(default_factory=lambda: [
        # Discrete values
        1, 2, 3, 5, 7,
        # Tuples with different dimensions
        (2, 2), (3, 3), (5, 5),
        (2, 3), (3, 2), (5, 7),
        # Boundary cases
        (1, 1), (1, 7), (7, 1)
    ])
    
    # stride: int or tuple of ints (height_stride, width_stride)
    stride: List[Union[int, Tuple[int, int]]] = field(default_factory=lambda: [
        # Discrete values
        1, 2, 3, 4, 5,
        # Tuples with different strides
        (1, 1), (2, 2), (3, 3),
        (2, 3), (3, 2), (4, 5),
        # Boundary cases
        (1, 5), (5, 1)
    ])
    
    # padding: int or tuple of ints (height_pad, width_pad)
    padding: List[Union[int, Tuple[int, int]]] = field(default_factory=lambda: [
        # Discrete values
        0, 1, 2, 3, 4,
        # Tuples with different padding
        (0, 0), (1, 1), (2, 2),
        (1, 2), (2, 1), (3, 4),
        # Boundary cases
        (0, 4), (4, 0)
    ])

# This ensures the class can be instantiated with var=InputSpace()
# Example usage:
# var = InputSpace()