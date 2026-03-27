import keras
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List

# Valid test case that ensures successful API call
valid_test_case = {
    'rate': 0.5,
    'data_format': 'channels_first',
    'seed': None,
    'name': None,
    'dtype': None,
    'inputs': np.random.randn(2, 4, 8, 8, 8).astype(np.float32),
    'training': True
}

@dataclass
class InputSpace:
    # Only parameter that affects output shape (except 'inputs')
    data_format: List[str] = field(default_factory=lambda: ['channels_first', 'channels_last'])