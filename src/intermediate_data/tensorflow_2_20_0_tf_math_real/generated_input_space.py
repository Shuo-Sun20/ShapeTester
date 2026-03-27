import tensorflow as tf
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional

def call_func(inputs, name=None):
    return tf.math.real(inputs, name=name)

# 1. valid_test_case definition
np.random.seed(42)
real_part = np.random.randn(3, 4).astype(np.float32)
imag_part = np.random.randn(3, 4).astype(np.float32)
complex_tensor = tf.constant(real_part + 1j * imag_part)

valid_test_case = {
    'inputs': complex_tensor,
    'name': 'test_real_operation'
}

# 2. Parameters that affect output shape (except 'inputs'): only 'name'
# 3. Value space analysis:
#    - 'name': discrete parameter (string/None), doesn't affect shape but is a parameter
#    Value space: [None, 'test_real_operation', 'real_part', 'my_real_op', 'another_name']

# 4. InputSpace definition
@dataclass
class InputSpace:
    name: List[Optional[str]] = field(default_factory=lambda: [
        None,
        'test_real_operation',
        'real_part',
        'my_real_op',
        'another_name'
    ])