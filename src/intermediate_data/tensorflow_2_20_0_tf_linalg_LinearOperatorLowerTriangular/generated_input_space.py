import tensorflow as tf
from dataclasses import dataclass, field
from typing import List, Union, Optional

# 1. Valid test case
valid_test_case = {
    'inputs': tf.random.normal(shape=[2, 4, 4]),
    'is_non_singular': None,
    'is_self_adjoint': False,
    'is_positive_definite': None,
    'is_square': True,
    'name': 'LinearOperatorLowerTriangular'
}

# 3. Define discretized value spaces for parameters
@dataclass
class InputSpace:
    """Contains all parameters that affect output shape with discretized value ranges"""
    
    # Boolean parameters with complete discrete value space
    is_non_singular: List[Optional[bool]] = field(default_factory=lambda: [None, True, False])
    is_self_adjoint: List[Optional[bool]] = field(default_factory=lambda: [None, True, False])
    is_positive_definite: List[Optional[bool]] = field(default_factory=lambda: [None, True, False])
    is_square: List[Optional[bool]] = field(default_factory=lambda: [None, True, False])
    
    # String parameter with typical values
    name: List[str] = field(default_factory=lambda: [
        'LinearOperatorLowerTriangular',
        'LowerTriangularOp',
        'TestOp',
        'CustomTriangular',
        'default'
    ])