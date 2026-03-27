import tensorflow as tf
import numpy as np
from dataclasses import dataclass

def call_func(inputs, name=None):
    return tf.nn.l2_loss(t=inputs, name=name)

# 1. Define valid_test_case
valid_test_case = {
    'inputs': tf.constant(np.random.randn(3, 4).astype(np.float32)),
    'name': 'test_l2_loss'
}

# 4. Define InputSpace
@dataclass
class InputSpace:
    name: list = None
    
    def __post_init__(self):
        if self.name is None:
            self.name = [None, 'l2_loss_1', 'l2_loss_2', 'l2_loss_3', 'l2_loss_4']