import tensorflow as tf
from dataclasses import dataclass, field
from typing import Optional, List

# Task 1: Define valid_test_case dictionary
valid_test_case = {
    'inputs': [tf.matmul(tf.random.normal(shape=(5, 5)), tf.random.normal(shape=(5, 5)), transpose_b=True) + tf.eye(5) * 0.1],
    'name': None
}

# Task 4: Define InputSpace dataclass
@dataclass
class InputSpace:
    """Contains parameters affecting output tensor shape for call_func"""
    
    # The only parameter (besides 'inputs') that can affect output shape is 'name'
    name: Optional[List[str]] = field(default_factory=lambda: [
        None,
        'test_logdet',
        'logdet_op',
        'matrix_logdet',
        'custom_name'
    ])