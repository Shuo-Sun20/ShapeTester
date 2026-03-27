import keras
from dataclasses import dataclass, field
from typing import Union, List, Tuple

def call_func(inputs, source, destination):
    return keras.ops.moveaxis(x=inputs, source=source, destination=destination)

# Task 1: Define valid_test_case
valid_test_case = {
    "inputs": keras.random.normal(shape=(3, 4, 5)),
    "source": 0,
    "destination": -1
}

# Tasks 2-4: Define InputSpace dataclass with discretized parameter spaces
@dataclass
class InputSpace:
    # Parameter affecting output shape: source
    source: List[Union[int, Tuple[int, ...], List[int]]] = field(
        default_factory=lambda: [
            # Single integer cases
            0, 1, -1, -2,
            # Multiple axes - length 2
            (0, 1), (0, -1), (1, -1), (-1, -2),
            # Multiple axes - length 3  
            (0, 1, 2), (2, 1, 0), (-1, -2, -3)
        ]
    )
    
    # Parameter affecting output shape: destination  
    destination: List[Union[int, Tuple[int, ...], List[int]]] = field(
        default_factory=lambda: [
            # Single integer cases
            0, 1, -1, -2,
            # Multiple axes - length 2
            (0, 1), (0, -1), (1, -1), (-1, -2),
            # Multiple axes - length 3
            (0, 1, 2), (2, 1, 0), (-1, -2, -3),
            # Boundary and special cases
            -3, 2, (0, 2, 1), (-3, -1, -2)
        ]
    )

# Example instantiation
var = InputSpace()