import tensorflow as tf
from dataclasses import dataclass, field
from typing import List, Any

# Define valid_test_case variable
valid_test_case = {
    "operators": [
        [tf.linalg.LinearOperatorFullMatrix(tf.random.normal(shape=[2, 2]))],
        [tf.linalg.LinearOperatorFullMatrix(tf.random.normal(shape=[2, 2])),
         tf.linalg.LinearOperatorFullMatrix(tf.random.normal(shape=[2, 2]))]
    ],
    "is_non_singular": None,
    "is_self_adjoint": None,
    "is_positive_definite": None,
    "is_square": None,
    "inputs": tf.random.normal(shape=[4, 3]),
    "adjoint": False,
    "adjoint_arg": False
}

@dataclass
class InputSpace:
    """
    Dataclass containing all parameters that affect the output tensor shape
    in LinearOperatorBlockLowerTriangular.matmul, with discretized value spaces.
    
    Notes on shape-affecting parameters:
    1. operators: Determines the shape of the LinearOperator itself.
    2. adjoint: When True, swaps domain and range dimensions.
    3. adjoint_arg: When True, transposes the input matrix.
    """
    
    # Parameter 1: operators - controls the block structure and dimensions
    operators: List[List[tf.linalg.LinearOperator]] = field(default_factory=lambda: [
        # 2x2 block structure (4x4 matrix)
        [
            [tf.linalg.LinearOperatorFullMatrix(tf.ones([2, 2]))],
            [tf.linalg.LinearOperatorFullMatrix(tf.ones([2, 2])),
             tf.linalg.LinearOperatorFullMatrix(tf.ones([2, 2]))]
        ],
        # 3x3 block structure (6x6 matrix) with [2,2,2] block sizes
        [
            [tf.linalg.LinearOperatorFullMatrix(tf.ones([2, 2]))],
            [tf.linalg.LinearOperatorFullMatrix(tf.ones([2, 2])),
             tf.linalg.LinearOperatorFullMatrix(tf.ones([2, 2]))],
            [tf.linalg.LinearOperatorFullMatrix(tf.ones([2, 2])),
             tf.linalg.LinearOperatorFullMatrix(tf.ones([2, 2])),
             tf.linalg.LinearOperatorFullMatrix(tf.ones([2, 2]))]
        ],
        # 2x2 block structure with different block sizes [3, 4] (7x7 matrix)
        [
            [tf.linalg.LinearOperatorFullMatrix(tf.ones([3, 3]))],
            [tf.linalg.LinearOperatorFullMatrix(tf.ones([4, 3])),
             tf.linalg.LinearOperatorFullMatrix(tf.ones([4, 4]))]
        ],
        # Single block (2x2 matrix)
        [
            [tf.linalg.LinearOperatorFullMatrix(tf.ones([2, 2]))]
        ],
        # 3x3 block structure with varying block sizes [1, 2, 3] (6x6 matrix)
        [
            [tf.linalg.LinearOperatorFullMatrix(tf.ones([1, 1]))],
            [tf.linalg.LinearOperatorFullMatrix(tf.ones([2, 1])),
             tf.linalg.LinearOperatorFullMatrix(tf.ones([2, 2]))],
            [tf.linalg.LinearOperatorFullMatrix(tf.ones([3, 1])),
             tf.linalg.LinearOperatorFullMatrix(tf.ones([3, 2])),
             tf.linalg.LinearOperatorFullMatrix(tf.ones([3, 3]))]
        ]
    ])
    
    # Parameter 2: adjoint - boolean flag for operator adjoint
    adjoint: List[bool] = field(default_factory=lambda: [
        False,    # Default case
        True,     # Adjoint operator
        # Note: Boundary values covered above
    ])
    
    # Parameter 3: adjoint_arg - boolean flag for input adjoint
    adjoint_arg: List[bool] = field(default_factory=lambda: [
        False,    # Default case
        True,     # Adjoint input
        # Note: Boundary values covered above
    ])