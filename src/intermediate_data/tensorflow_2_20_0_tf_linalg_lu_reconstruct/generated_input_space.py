import tensorflow as tf
import numpy as np
from dataclasses import dataclass, field
from typing import Union, List, Optional

# Define valid_test_case with all call_func parameters
def call_func(inputs, validate_args=False, name=None):
    lower_upper, perm = inputs
    return tf.linalg.lu_reconstruct(lower_upper=lower_upper, perm=perm, validate_args=validate_args, name=name)

# Generate example inputs
x = tf.random.normal(shape=(2, 3, 3))
lu, perm = tf.linalg.lu(x)

# valid_test_case containing all call_func parameters
valid_test_case = {
    'inputs': [lu, perm],
    'validate_args': False,
    'name': None
}

# Dataclass for InputSpace
@dataclass
class InputSpace:
    """
    Contains parameters that affect the shape of the output tensor from tf.linalg.lu_reconstruct.
    
    Note: Only the shapes of tensors in 'inputs' affect output shape.
    Parameters 'validate_args' and 'name' do not affect output shape.
    """
    
    # The 'inputs' parameter affects output shape but is excluded per instructions.
    # 'validate_args' and 'name' are included with their discrete value spaces.
    
    # validate_args: Boolean parameter that controls argument validation
    validate_args: List[bool] = field(default_factory=lambda: [False, True])
    
    # name: String identifier for the operation
    name: List[Optional[str]] = field(default_factory=lambda: [None, 'lu_reconstruct_op', 'custom_name_1', 'custom_name_2', 'custom_name_3'])