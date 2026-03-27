import numpy as np
from dataclasses import dataclass, field

valid_test_case = {
    'data_format': 'channels_last',
    'keepdims': False,
    'inputs': np.random.rand(2, 4, 5, 4, 3).astype(np.float32)
}

@dataclass
class InputSpace:
    data_format: list = field(default_factory=lambda: ['channels_last', 'channels_first'])
    keepdims: list = field(default_factory=lambda: [True, False])