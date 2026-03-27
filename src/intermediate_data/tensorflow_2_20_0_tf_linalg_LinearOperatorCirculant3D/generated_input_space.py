import tensorflow as tf
from dataclasses import dataclass
from typing import Optional

valid_test_case = {
    'inputs': tf.complex(
        tf.random.normal([2, 3, 4], dtype=tf.float32),
        tf.random.normal([2, 3, 4], dtype=tf.float32)
    ),
    'is_non_singular': None,
    'is_self_adjoint': None,
    'is_positive_definite': None,
    'is_square': None,
    'name': 'LinearOperatorCirculant3D'
}

@dataclass
class InputSpace:
    """
    Value ranges for parameters affecting output tensor shape.
    Note: Only the 'inputs' parameter affects output shape, but it's excluded
    per requirements. Other parameters (hints) don't affect shape but are
    included for completeness with discretized value ranges.
    """
    is_non_singular: Optional[list[Optional[bool]]] = None
    is_self_adjoint: Optional[list[Optional[bool]]] = None
    is_positive_definite: Optional[list[Optional[bool]]] = None
    is_square: Optional[list[Optional[bool]]] = None
    name: Optional[list[str]] = None

    def __post_init__(self):
        # Boolean hint parameters: discrete values (True, False, None)
        if self.is_non_singular is None:
            self.is_non_singular = [True, False, None]
        if self.is_self_adjoint is None:
            self.is_self_adjoint = [True, False, None]
        if self.is_positive_definite is None:
            self.is_positive_definite = [True, False, None]
        if self.is_square is None:
            self.is_square = [True, False, None]
        # String name parameter: example values
        if self.name is None:
            self.name = [
                'LinearOperatorCirculant3D',
                'Circulant3D_Op',
                'Test_Operator',
                'Custom_Circulant3D',
                'Default'
            ]