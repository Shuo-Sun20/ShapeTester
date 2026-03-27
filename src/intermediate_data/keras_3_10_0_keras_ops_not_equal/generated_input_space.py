import keras
import numpy as np
from dataclasses import dataclass, field
from typing import List, Union

# Task 1: Define valid_test_case
valid_test_case = {
    "inputs": [
        keras.ops.convert_to_tensor(np.random.rand(3, 4)),
        keras.ops.convert_to_tensor(np.random.rand(3, 4))
    ]
}

# Tasks 2-4: Define InputSpace dataclass
@dataclass
class InputSpace:
    # Parameters affecting output shape (except 'inputs'): none
    # The shape is determined by the x1, x2 tensors inside 'inputs'
    # We'll create a helper class for the tensors inside inputs
    pass

# Since the actual shape-affecting parameters are x1 and x2 within the inputs list,
# we need to define their value spaces separately
class TensorValueSpace:
    # Define shape possibilities for tensors
    shapes = [
        (3, 4),           # Same shape as example
        (3, 1),           # Broadcastable shape
        (1, 4),           # Broadcastable shape
        (3,),             # 1D tensor
        (1,),             # Scalar-like
        (5, 3, 4),        # Higher dimension
        (0, 4),           # Empty dimension
    ]
    
    # Define dtype possibilities
    dtypes = [
        np.float32,
        np.float64,
        np.int32,
        np.int64,
        np.bool_,
    ]
    
    # Define value ranges for continuous parameters
    # Using discretized values for demonstration
    value_ranges = [
        np.array([0.0]),                  # All zeros
        np.array([1.0]),                  # All ones
        np.array([0.0, 0.25, 0.5, 0.75, 1.0]),  # Range of values
        np.array([-1.0, -0.5, 0.0, 0.5, 1.0]),  # Signed range
        np.array([0.0, 1.0, 2.0, 3.0, 4.0]),    # Integer-like floats
    ]