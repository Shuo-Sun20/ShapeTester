import tensorflow as tf
from dataclasses import dataclass, field
from typing import List, Optional, Union, Tuple

# 1. Valid test case
valid_test_case = {
    'inputs': [
        tf.random.normal(shape=(2, 3)), 
        tf.random.normal(shape=(4,))
    ],
    'name': None
}

# 3. Parameter value spaces:
# - 'name': string or None, doesn't affect output shape
#   Discretized to 5 values including None
name_values = [None, 'norm1', 'norm2', 'global_norm', 'test']

@dataclass
class InputSpace:
    name: List[Optional[str]] = field(default_factory=lambda: name_values)