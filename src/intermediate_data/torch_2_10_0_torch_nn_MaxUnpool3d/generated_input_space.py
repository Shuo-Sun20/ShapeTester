import torch
from dataclasses import dataclass, field
from typing import Union, Tuple, List

# 1. Define valid_test_case
valid_test_case = {
    'kernel_size': 3,
    'stride': 2,
    'padding': 0,
    'inputs': [torch.randn(20, 16, 51, 33, 15), None, None]  # Placeholder, will be replaced
}

# Create actual input for the test case
pool = torch.nn.MaxPool3d(kernel_size=3, stride=2, return_indices=True)
input_tensor = torch.randn(20, 16, 51, 33, 15)
output, indices = pool(input_tensor)
valid_test_case['inputs'] = [output, indices]

# 2. & 3. Identify shape-affecting parameters and their value spaces
# The parameters that affect output shape are: kernel_size, stride, padding
# All can be int or tuple of ints (3-tuple for 3D)

@dataclass
class InputSpace:
    """Dataclass containing all parameters affecting output shape with discretized value ranges"""
    
    # kernel_size: int or tuple of 3 ints (min 1)
    kernel_size: List[Union[int, Tuple[int, int, int]]] = field(
        default_factory=lambda: [
            1, 2, 3, 5, 7,  # Single int values
            (1, 1, 1), (2, 2, 2), (3, 3, 3), (5, 5, 5), (7, 7, 7),  # Tuple values
            (1, 3, 5), (7, 5, 3)  # Asymmetric tuples
        ]
    )
    
    # stride: int or tuple of 3 ints (min 1)
    stride: List[Union[int, Tuple[int, int, int]]] = field(
        default_factory=lambda: [
            1, 2, 3, 4, 5,  # Single int values
            (1, 1, 1), (2, 2, 2), (3, 3, 3), (4, 4, 4), (5, 5, 5),  # Tuple values
            (1, 2, 3), (4, 3, 2)  # Asymmetric tuples
        ]
    )
    
    # padding: int or tuple of 3 ints (min 0)
    padding: List[Union[int, Tuple[int, int, int]]] = field(
        default_factory=lambda: [
            0, 1, 2, 3, 4,  # Single int values
            (0, 0, 0), (1, 1, 1), (2, 2, 2), (3, 3, 3), (4, 4, 4),  # Tuple values
            (0, 1, 2), (3, 2, 1)  # Asymmetric tuples
        ]
    )