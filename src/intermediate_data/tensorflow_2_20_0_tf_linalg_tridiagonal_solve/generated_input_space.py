import tensorflow as tf
from dataclasses import dataclass, field
from typing import List, Optional

# The original call_func definition
def call_func(inputs, diagonals_format='compact', transpose_rhs=False, 
              conjugate_rhs=False, name=None, partial_pivoting=True, 
              perturb_singular=False):
    diagonals = inputs[0]
    rhs = inputs[1]
    return tf.linalg.tridiagonal_solve(
        diagonals=diagonals,
        rhs=rhs,
        diagonals_format=diagonals_format,
        transpose_rhs=transpose_rhs,
        conjugate_rhs=conjugate_rhs,
        name=name,
        partial_pivoting=partial_pivoting,
        perturb_singular=perturb_singular
    )

# Example tensors
M = 5
K = 2
batch_size = 3
compact_diagonals = tf.random.normal(shape=[batch_size, 3, M])
rhs = tf.random.normal(shape=[batch_size, M, K])
inputs = [compact_diagonals, rhs]

# 1. valid_test_case dictionary
valid_test_case = {
    'inputs': inputs,
    'diagonals_format': 'compact',
    'transpose_rhs': False,
    'conjugate_rhs': False,
    'name': None,
    'partial_pivoting': True,
    'perturb_singular': False
}

# 2. Parameters affecting output shape (excluding 'inputs'):
#    - transpose_rhs: Affects whether rhs is transposed before solving
#    - conjugate_rhs: Affects conjugation of rhs (but not shape)

# 3. Discretized value spaces:
#    transpose_rhs: boolean [True, False]
#    conjugate_rhs: boolean [True, False]

# 4. InputSpace dataclass
@dataclass
class InputSpace:
    # Parameters affecting output shape (excludes 'inputs')
    transpose_rhs: List[bool] = field(default_factory=lambda: [True, False])
    conjugate_rhs: List[bool] = field(default_factory=lambda: [True, False])
    
    # Other parameters that don't affect shape but are in call_func
    diagonals_format: List[str] = field(default_factory=lambda: ['matrix', 'sequence', 'compact'])
    partial_pivoting: List[bool] = field(default_factory=lambda: [True, False])
    perturb_singular: List[bool] = field(default_factory=lambda: [True, False])
    name: List[Optional[str]] = field(default_factory=lambda: [None])

# The class can be instantiated as: var = InputSpace()