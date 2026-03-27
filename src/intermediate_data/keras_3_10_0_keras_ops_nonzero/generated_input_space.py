import keras
import numpy as np
from dataclasses import dataclass, field
from typing import List

# 1. Define valid_test_case
np.random.seed(42)
random_tensor = keras.ops.convert_to_tensor(
    np.random.randint(0, 3, size=(3, 3)).astype(np.float32)
)
valid_test_case = {"inputs": [random_tensor]}

# 2. Parameters affecting output shape: only 'inputs' (contains the tensor)
# 3. Value space analysis for the input tensor

@dataclass
class InputSpace:
    # The only parameter that affects output shape
    inputs: List[keras.KerasTensor] = field(default_factory=list)
    
    def __post_init__(self):
        if not self.inputs:
            self._generate_inputs()
    
    def _generate_inputs(self):
        """Generate representative input tensors covering various cases"""
        tensors = []
        
        # Start with the valid test case tensor
        tensors.append(valid_test_case["inputs"][0])
        
        # Discrete shapes covering different ranks and dimensions
        # Rank 0: scalar
        tensors.append(keras.ops.convert_to_tensor(5.0))
        
        # Rank 1: vectors
        shapes_1d = [(0,), (1,), (5,), (10,), (100,)]
        for shape in shapes_1d:
            arr = np.random.randn(*shape).astype(np.float32)
            tensors.append(keras.ops.convert_to_tensor(arr))
        
        # Rank 2: matrices
        shapes_2d = [(0, 0), (1, 1), (3, 3), (5, 5), (10, 10)]
        for shape in shapes_2d:
            arr = np.random.randn(*shape).astype(np.float32)
            tensors.append(keras.ops.convert_to_tensor(arr))
        
        # Rank 3: 3D tensors
        shapes_3d = [(2, 2, 2), (3, 3, 3), (4, 4, 4)]
        for shape in shapes_3d:
            arr = np.random.randn(*shape).astype(np.float32)
            tensors.append(keras.ops.convert_to_tensor(arr))
        
        # Rank 4: 4D tensors
        shapes_4d = [(2, 2, 2, 2), (3, 3, 3, 3)]
        for shape in shapes_4d:
            arr = np.random.randn(*shape).astype(np.float32)
            tensors.append(keras.ops.convert_to_tensor(arr))
        
        # Various sparsity patterns
        # All zeros
        all_zeros = keras.ops.zeros((5, 5))
        tensors.append(all_zeros)
        
        # All non-zeros
        all_nonzeros = keras.ops.ones((5, 5))
        tensors.append(all_nonzeros)
        
        # Sparse pattern
        sparse_arr = np.zeros((5, 5), dtype=np.float32)
        sparse_arr[0, 0] = 1.0
        sparse_arr[2, 2] = 1.0
        sparse_arr[4, 4] = 1.0
        tensors.append(keras.ops.convert_to_tensor(sparse_arr))
        
        # Boundary: very large shape (but not too large for memory)
        large_arr = np.random.randn(100, 100).astype(np.float32)
        tensors.append(keras.ops.convert_to_tensor(large_arr))
        
        # Wrap each tensor in a list as expected by call_func
        self.inputs = [[tensor] for tensor in tensors]