import keras
import tensorflow as tf
from dataclasses import dataclass, field
from typing import Tuple, Optional

# 1. Define a valid test case
random_tensor = tf.random.uniform(shape=(1, 28, 28, 3), minval=0, maxval=256, dtype=tf.float32)
valid_test_case = {
    'inputs': random_tensor,
    'value_range': (0, 255),
    'bins': 256,
    'data_format': None
}

# 2 & 3. Identify shape-affecting parameters and their value spaces
# Parameters that can affect output shape: data_format
# value_range and bins only affect intensity distribution, not tensor shape

@dataclass
class InputSpace:
    """
    Dataclass containing all parameters that can affect the output tensor shape
    of the Equalization layer.
    """
    # data_format can be None, 'channels_last', or 'channels_first'
    # This affects the output shape by determining dimension ordering
    data_format: Optional[str] = field(default_factory=lambda: [None, 'channels_last', 'channels_first'])
    
    # Note: value_range and bins are excluded as they don't affect tensor shape