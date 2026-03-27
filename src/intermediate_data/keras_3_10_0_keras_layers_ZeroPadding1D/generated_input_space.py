import numpy as np
import keras
from dataclasses import dataclass, field
from typing import Union, List, Tuple, Optional

# 1. Define valid_test_case
valid_test_case = {
    'inputs': np.random.randn(2, 5, 3),  # Shape (batch_size, axis_to_pad, features)
    'padding': 2,
    'data_format': 'channels_last'
}

# 2. Parameters affecting output shape (excluding inputs): 'padding', 'data_format'

# 3. Value space analysis and 4. InputSpace dataclass definition
@dataclass
class InputSpace:
    padding: List[Union[int, Tuple[int, int]]] = field(
        default_factory=lambda: [0, 1, 2, (0, 1), (1, 0)]
    )
    data_format: List[Optional[str]] = field(
        default_factory=lambda: ['channels_last', 'channels_first', None]
    )