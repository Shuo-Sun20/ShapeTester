import tensorflow as tf
import numpy as np
from dataclasses import dataclass
from typing import List, Optional

def call_func(inputs, name=None):
    return tf.nn.l2_loss(t=inputs, name=name)

# 1. Define valid_test_case
valid_test_case = {
    'inputs': tf.constant(np.random.randn(3, 4).astype(np.float32)),
    'name': 'test_l2_loss'
}

# 2. Identify parameters affecting output shape (except "inputs")
# For tf.nn.l2_loss, only the 'name' parameter is in the parameter list besides 'inputs'
# However, the output of tf.nn.l2_loss is always a scalar (0-D tensor) regardless of input shape
# So technically, no parameters affect the output shape, but 'name' is a parameter in the function

# 3. Construct value space for 'name' parameter
# 'name' can be None or any valid string
# Since it doesn't affect the output shape but is a parameter, we'll create a discretized value space

# 4. Define InputSpace class
@dataclass
class InputSpace:
    # 'name' is the only parameter in call_func besides 'inputs'
    # The value space includes boundary values (None) and various string examples
    name: List[Optional[str]] = None
    
    def __post_init__(self):
        if self.name is None:
            # Discretized value space for name parameter
            self.name = [
                None,  # boundary value: no name
                'test_l2_loss',  # value from valid_test_case
                'l2_loss_op',   # typical value
                'my_l2_loss',   # typical value
                'loss_calculation',  # typical value
                '',  # boundary value: empty string
                'l2_loss_1'  # typical value
            ]

# Note: The InputSpace class is defined to contain parameters that affect the shape,
# but tf.nn.l2_loss always outputs a scalar (shape []) regardless of input,
# so 'name' doesn't affect the output shape. However, it's included as it's 
# the only parameter in call_func besides 'inputs'.