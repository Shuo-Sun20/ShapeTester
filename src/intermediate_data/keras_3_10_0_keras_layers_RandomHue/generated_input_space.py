import numpy as np
import keras
from dataclasses import dataclass, field
from typing import Tuple, Optional, List, Union

# Task 1: Define valid_test_case
example_input = np.random.rand(8, 32, 32, 3).astype('float32')
valid_test_case = {
    'inputs': example_input,
    'factor': 0.5,
    'value_range': [0, 1],
    'data_format': None,
    'seed': None
}

# Task 2: Identify parameters affecting output shape
# From RandomHue documentation and Keras knowledge:
# - inputs: affects shape (but excluded per instructions)
# - data_format: affects shape interpretation (channels position)
# - Other parameters (factor, value_range, seed) don't affect tensor shape

# Task 3: Discretize parameter value spaces
# data_format: discrete parameter with possible values
data_format_values = [None, 'channels_last', 'channels_first']

# Task 4: Define InputSpace dataclass
@dataclass
class InputSpace:
    data_format: List[Optional[str]] = field(default_factory=lambda: [None, 'channels_last', 'channels_first'])

# Example instantiation
var = InputSpace()