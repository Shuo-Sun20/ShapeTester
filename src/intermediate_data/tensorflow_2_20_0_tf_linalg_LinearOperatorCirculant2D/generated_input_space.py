import tensorflow as tf
from dataclasses import dataclass
from typing import Optional, Union, List

# Define tensors for valid test case
spectrum = tf.complex(
    tf.random.normal([2, 3], dtype=tf.float32),
    tf.random.normal([2, 3], dtype=tf.float32)
)
vector = tf.complex(
    tf.random.normal([6], dtype=tf.float32),
    tf.random.normal([6], dtype=tf.float32)
)

valid_test_case = {
    'inputs': [spectrum, vector],
    'input_output_dtype': None,
    'is_non_singular': None,
    'is_self_adjoint': None,
    'is_positive_definite': None,
    'is_square': True,
    'name': "LinearOperatorCirculant2D"
}

@dataclass
class InputSpace:
    input_output_dtype: List[Optional[tf.DType]] = None
    is_square: List[Optional[bool]] = None
    
    def __post_init__(self):
        # For input_output_dtype: None means use spectrum.dtype, otherwise can be tf.complex64/tf.complex128
        if self.input_output_dtype is None:
            self.input_output_dtype = [None, tf.complex64, tf.complex128]
        
        # For is_square: According to documentation, LinearOperatorCirculant2D is always square
        # but the parameter accepts True/False/None
        if self.is_square is None:
            self.is_square = [True, False, None]

# Initialize with default value ranges
input_space = InputSpace()