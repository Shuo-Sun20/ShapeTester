import tensorflow as tf
from dataclasses import dataclass, field

# 1. Define a valid test case dictionary
data = tf.constant([[1, 2, 3, 4], [5, 6, 7, 8], [4, 3, 2, 1]])
segment_ids = tf.constant([0, 1, 0])
valid_test_case = {
    'inputs': [data, segment_ids],
    'num_segments': 2
}

# 2. Parameters affecting output shape (excluding 'inputs'): num_segments
# 3. Parameter type analysis:
#    - num_segments: discrete positive integer (int32/int64)
#    Value space includes boundary and typical values

# 4. Define InputSpace class with shape-affecting parameters
@dataclass
class InputSpace:
    # num_segments affects the first dimension of output shape
    num_segments: list = field(default_factory=lambda: [0, 1, 2, 3, 4])