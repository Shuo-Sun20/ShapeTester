import keras
import numpy as np
from dataclasses import dataclass, field
from typing import List, Union, Any

def call_func(subscripts, inputs):
    if isinstance(subscripts, str):
        return keras.ops.einsum(subscripts, *inputs)
    else:
        args = []
        for i, inp in enumerate(inputs):
            args.append(inp)
            args.append(subscripts[i])
        if len(subscripts) > len(inputs):
            args.append(subscripts[-1])
        return keras.ops.einsum(*args)

# 1. Valid test case
matrix1 = keras.random.normal(shape=(3, 4))
matrix2 = keras.random.normal(shape=(4, 5))
valid_test_case = {
    "subscripts": "ij,jk->ik",
    "inputs": [matrix1, matrix2]
}

# 2. & 3. Parameters affecting output shape (excluding "inputs") and their value spaces
# The only parameter is "subscripts" which can be:
# - String format: "subscripts->output" or implicit format
# - List format: [operand1_indices, operand2_indices, ..., output_indices]

# Value space for subscripts parameter
string_formats = [
    # Basic operations
    "ii",                    # Trace (scalar output)
    "ii->i",                 # Diagonal (1D output)
    "ij->i",                 # Row sum (1D output)
    "ij->j",                 # Column sum (1D output)
    "ij->ij",                # Identity (2D output)
    "ij->ji",                # Transpose (2D output)
    
    # Two-operand operations
    "ij,jk->ik",            # Matrix multiplication (2D output)
    "ij,j->i",              # Matrix-vector multiplication (1D output)
    "i,i->",                # Dot product (scalar output)
    "i,j->ij",              # Outer product (2D output)
    "ij,ij->ij",            # Element-wise multiplication (2D output)
    
    # Three-operand operations
    "ij,jk,kl->il",         # Chain of matrix multiplications (2D output)
    "ijk,jkl->il",          # Batch matrix multiplication (3D output)
    
    # Higher dimensions
    "ijk->i",               # Sum over last two axes (1D output)
    "ijk->j",               # Sum over first and last axes (1D output)
    "ijk->k",               # Sum over first two axes (1D output)
    "ijk->ij",              # Sum over last axis (2D output)
    "ijk->ik",              # Sum over middle axis (2D output)
    "ijk->jk",              # Sum over first axis (2D output)
    
    # Ellipsis notation
    "...ii->...i",          # Batched diagonal
    "...ij->...i",          # Batched row sum
    "...ij,...jk->...ik",   # Batched matrix multiplication
    
    # Edge cases
    "i->i",                 # Identity for vector
    "->",                   # Constant (scalar output)
    "i->",                  # Vector sum (scalar output)
]

list_formats = [
    # Corresponding to string formats
    [[0, 0]],                          # Trace
    [[0, 0], [0]],                     # Diagonal
    [[0, 1], [0]],                     # Row sum
    [[0, 1], [1]],                     # Column sum
    [[0, 1], [0, 1]],                  # Identity
    [[0, 1], [1, 0]],                  # Transpose
    
    # Two operands
    [[0, 1], [1, 2], [0, 2]],         # Matrix multiplication
    [[0, 1], [1], [0]],               # Matrix-vector multiplication
    [[0], [0], []],                   # Dot product
    [[0], [1], [0, 1]],               # Outer product
    [[0, 1], [0, 1], [0, 1]],         # Element-wise multiplication
    
    # Three operands
    [[0, 1], [1, 2], [2, 3], [0, 3]], # Chain multiplication
    [[0, 1, 2], [1, 2, 3], [0, 3]],   # Batch matrix multiplication
    
    # Higher dimensions
    [[0, 1, 2], [0]],                 # Sum over last two axes
    [[0, 1, 2], [1]],                 # Sum over first and last
    [[0, 1, 2], [2]],                 # Sum over first two
    [[0, 1, 2], [0, 1]],              # Sum over last axis
    [[0, 1, 2], [0, 2]],              # Sum over middle axis
    [[0, 1, 2], [1, 2]],              # Sum over first axis
    
    # Edge cases
    [[0], [0]],                       # Identity for vector
    [[]],                             # Constant
    [[0], []],                        # Vector sum
]

# Combine all subscript formats
subscripts_value_space = string_formats + list_formats

# 4. InputSpace dataclass
@dataclass
class InputSpace:
    subscripts: List[Union[str, List[List[int]]]] = field(
        default_factory=lambda: subscripts_value_space
    )