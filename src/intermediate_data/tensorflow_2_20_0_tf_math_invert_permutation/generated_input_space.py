import tensorflow as tf
import numpy as np
from dataclasses import dataclass, field
from typing import List, Union, Optional

# 1. Define valid_test_case
valid_test_case = {
    'inputs': tf.constant([3, 4, 0, 2, 1], dtype=tf.int32),
    'name': 'invert_permutation_example'
}

# 2. Identify parameters that affect output shape (except 'inputs')
# Only 'name' remains, but it does NOT affect output shape
# Output shape is solely determined by input tensor shape

# 3. Since no parameters affect output shape except 'inputs' (which is excluded),
# InputSpace will have no fields

@dataclass
class InputSpace:
    # No fields required as no parameters affect output shape except 'inputs'
    pass