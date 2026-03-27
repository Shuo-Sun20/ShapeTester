import tensorflow as tf
from dataclasses import dataclass, field
from typing import Optional, Union

# 1. Define valid_test_case variable
valid_test_case = {
    "inputs": [tf.random.normal(shape=[3, 4]), tf.random.normal(shape=[3, 4, 2])],
    "is_non_singular": True,
    "is_self_adjoint": True,
    "is_positive_definite": True,
    "is_square": True,
    "name": "test_operator"
}

# 2. Parameters affecting output shape: is_square
# Only is_square parameter affects the shape of LinearOperatorDiag and thus the matmul output

# 3. Parameter type analysis and discretized value spaces:
# - is_square: boolean, possible values [True, False]
# - is_non_singular: boolean or None, values [None, True, False]
# - is_self_adjoint: boolean or None, values [None, True, False]
# - is_positive_definite: boolean or None, values [None, True, False]
# - name: string or None, values [None, "op1", "op2", "op3", "custom_name"]

# 4. Define InputSpace dataclass with all parameters that affect output shape
@dataclass
class InputSpace:
    is_square: list = field(default_factory=lambda: [True, False])
    # Note: Although only is_square directly affects output shape, 
    # other parameters are included for completeness as they're part of the API
    is_non_singular: list = field(default_factory=lambda: [None, True, False])
    is_self_adjoint: list = field(default_factory=lambda: [None, True, False])
    is_positive_definite: list = field(default_factory=lambda: [None, True, False])
    name: list = field(default_factory=lambda: [None, "op1", "op2", "op3", "custom_name"])