import tensorflow as tf
from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Union
import numpy as np

valid_test_case = {
    'inputs': tf.random.normal(shape=[2, 3, 3, 4]),  # For 'compact' format
    'diagonals_format': 'compact',
    'is_non_singular': None,
    'is_self_adjoint': None,
    'is_positive_definite': None,
    'is_square': None,
    'name': 'LinearOperatorTridiag_test'
}

@dataclass
class InputSpace:
    """Class containing all parameters that affect the output shape of call_func"""
    
    diagonals_format: List[str] = field(
        default_factory=lambda: ['sequence', 'compact', 'matrix']
    )
    
    # Note: The following parameters do NOT affect output shape, but are included for completeness
    # as they are part of call_func's signature
    is_non_singular: List[Optional[bool]] = field(
        default_factory=lambda: [None, True, False]
    )
    is_self_adjoint: List[Optional[bool]] = field(
        default_factory=lambda: [None, True, False]
    )
    is_positive_definite: List[Optional[bool]] = field(
        default_factory=lambda: [None, True, False]
    )
    is_square: List[Optional[bool]] = field(
        default_factory=lambda: [None, True, False]
    )
    name: List[Optional[str]] = field(
        default_factory=lambda: [None, 'test_operator', 'custom_name']
    )