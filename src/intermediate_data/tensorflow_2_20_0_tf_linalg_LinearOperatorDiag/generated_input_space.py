import tensorflow as tf
from dataclasses import dataclass, field
from typing import Optional, Union, List

# 1. Define valid_test_case variable
valid_test_case = {
    "inputs": [tf.random.normal(shape=[3, 4]), tf.random.normal(shape=[3, 4, 2])],
    "is_non_singular": True,
    "is_self_adjoint": True,
    "is_positive_definite": True,
    "is_square": True,
    "name": "test_operator"
}

# 2. Parameters affecting output shape: 
#    - is_square: When False, operator may be non-square affecting matmul compatibility
#    - name: Doesn't affect output shape, but included per feedback requirements

# 3. Value space definitions
@dataclass
class InputSpace:
    is_non_singular: List[Optional[bool]] = field(
        default_factory=lambda: [True, False, None]
    )
    is_self_adjoint: List[Optional[bool]] = field(
        default_factory=lambda: [True, False, None]
    )
    is_positive_definite: List[Optional[bool]] = field(
        default_factory=lambda: [True, False, None]
    )
    is_square: List[Optional[bool]] = field(
        default_factory=lambda: [True, False, None]
    )
    name: List[Optional[str]] = field(
        default_factory=lambda: ["test_operator", "diag_op", None, "linear_operator", ""]
    )