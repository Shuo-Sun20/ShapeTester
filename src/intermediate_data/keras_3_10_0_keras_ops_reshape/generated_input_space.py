import keras
import numpy as np
from dataclasses import dataclass, field
from typing import Union, Tuple, List

def call_func(inputs, newshape):
    return keras.ops.reshape(inputs, newshape)

# Create random input tensor
random_tensor = keras.random.normal(shape=(6, 8))

# 1. Define valid_test_case
valid_test_case = {
    "inputs": random_tensor,
    "newshape": (12, 4)
}

# 2. Parameters affecting output shape (except "inputs"): newshape
# 3. Parameter type analysis and value space construction:
#    newshape: Can be integer, tuple of integers, or list of integers
#    Must be compatible with input tensor's total elements
#    For input shape (6,8) -> 48 elements
#    Represent legal shapes through combinations and -1 usage

@dataclass
class InputSpace:
    # 4. All parameters affecting output shape with discretized value ranges
    newshape: List[Union[int, Tuple[int, ...]]] = field(
        default_factory=lambda: [
            # Integer (1D) representations
            48,
            -1,
            
            # Tuple representations (2D)
            (48, 1),
            (1, 48),
            (24, 2),
            (2, 24),
            (12, 4),
            (4, 12),
            (8, 6),
            (6, 8),
            (16, 3),
            (3, 16),
            
            # Tuple representations (3D)
            (4, 6, 2),
            (3, 4, 4),
            (2, 3, 8),
            (2, 4, 6),
            (2, 2, 12),
            
            # With -1 inference (2D)
            (-1, 8),
            (6, -1),
            (-1, 4),
            (12, -1),
            (-1, 2),
            (24, -1),
            
            # With -1 inference (3D)
            (2, -1, 4),
            (-1, 3, 4),
            (2, 3, -1),
            
            # Edge cases
            (1, 1, 48),
            (48, 1, 1),
            (1, 48, 1),
        ]
    )

# Test instantiation
if __name__ == "__main__":
    # Test the valid test case
    result = call_func(**valid_test_case)
    print(f"Test case output shape: {result.shape}")
    
    # Test InputSpace instantiation
    space = InputSpace()
    print(f"Total newshape variations: {len(space.newshape)}")
    print(f"Sample newshape values: {space.newshape[:5]}")