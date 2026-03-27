import tensorflow as tf
from dataclasses import dataclass, field
from typing import List, Union

# 1. Define valid_test_case dictionary
superdiag = tf.random.uniform(shape=[5], dtype=tf.float32)
maindiag = tf.random.uniform(shape=[5], dtype=tf.float32)
subdiag = tf.random.uniform(shape=[5], dtype=tf.float32)
diagonals = [superdiag, maindiag, subdiag]
rhs = tf.random.uniform(shape=[5, 3], dtype=tf.float32)

valid_test_case = {
    'inputs': [diagonals, rhs],
    'diagonals_format': 'sequence',
    'name': None
}

# 2. & 3. & 4. Define InputSpace dataclass
@dataclass
class InputSpace:
    diagonals_format: List[str] = field(
        default_factory=lambda: ['sequence', 'compact']
    )
    
    name: List[Union[str, None]] = field(
        default_factory=lambda: [None, 'test_op', 'matmul_op', 'tridiag_mul', '']
    )