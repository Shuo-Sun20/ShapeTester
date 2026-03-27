import tensorflow as tf
from dataclasses import dataclass
from typing import Optional, List

# 1. Define valid_test_case
value = tf.random.normal(shape=(2, 3, 4, 5))  # 4D tensor with channel last
bias = tf.random.normal(shape=(5,))  # 1D tensor matching channel dimension
valid_test_case = {
    "inputs": [value, bias],
    "data_format": None,  # Optional parameter, can be None, 'N...C', or 'NC...'
    "name": None  # Optional operation name
}

# 2. Parameters affecting output shape (excluding "inputs"):
#    - data_format: Affects which dimension is considered the channel dimension
#    - Note: 'name' parameter does not affect output shape

# 3. Value space analysis for parameters affecting shape:
#    - data_format: Discrete parameter with possible values: [None, 'N...C', 'NC...']
#    - 'None' defaults to 'N...C' (channel last)

# 4. Define InputSpace dataclass
@dataclass
class InputSpace:
    # Only parameter affecting output shape (other than inputs)
    data_format: List[Optional[str]] = None
    
    def __post_init__(self):
        # Set default value space for data_format if not provided
        if self.data_format is None:
            self.data_format = [None, 'N...C', 'NC...']