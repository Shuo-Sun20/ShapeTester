import keras
from dataclasses import dataclass
from typing import Tuple, List, Union, Optional

# 1. Define a valid test case
valid_test_case = {
    "inputs": keras.ops.zeros(shape=(2, 3)),  # unused but must be a valid tensor
    "shape": (2, 3),  # directly affects output shape
    "dtype": "float32"  # doesn't affect shape
}

# 2. Parameters affecting output shape (excluding "inputs"):
#    - shape (only parameter that directly determines output tensor shape)

# 3. Value space analysis:
#    shape: tuple of ints (can be 0-5D, each dimension >= 0)
#    Boundary values: (0,), (0,0), (0,0,0), (1,), (1,1,1,1,1)
#    Typical values: (1,), (2,3), (4,5,6), (2,), (2,2,2)

# 4. InputSpace dataclass
@dataclass
class InputSpace:
    # Only parameter affecting output tensor shape
    shape: List[Tuple[int, ...]] = None
    
    def __post_init__(self):
        if self.shape is None:
            # Discretized value space for shape parameter
            # Covering 0D to 5D tensors with boundary and typical values
            self.shape = [
                # 0D tensor (scalar-like, but in Keras this is actually 1D with 1 element)
                (1,),
                
                # 1D tensors
                (0,),      # boundary: empty tensor
                (1,),      # boundary: single element
                (2,),      # typical
                (5,),      # typical
                (10,),     # typical
                
                # 2D tensors
                (0, 0),    # boundary: empty
                (1, 1),    # boundary: single element
                (2, 3),    # typical (from valid_test_case)
                (4, 5),    # typical
                (10, 10),  # typical
                
                # 3D tensors
                (0, 0, 0),  # boundary: empty
                (1, 1, 1),  # boundary: single element
                (2, 3, 4),  # typical
                (3, 3, 3),  # typical
                (5, 6, 7),  # typical
                
                # 4D tensors (e.g., batches of images)
                (2, 3, 4, 5),  # typical
                (1, 28, 28, 1),  # typical MNIST-like shape
                
                # 5D tensors (e.g., video data)
                (2, 3, 4, 5, 6),  # typical
                
                # Mixed dimensions with zeros
                (0, 5),    # partially empty
                (5, 0),    # partially empty
            ]

# Instantiation example
var = InputSpace()