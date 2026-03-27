import numpy as np
import keras
from dataclasses import dataclass, field
from typing import List

valid_test_case = {
    'num_tokens': 5,
    'output_mode': 'one_hot',
    'sparse': False,
    'inputs': np.random.randint(0, 5, size=(10,))
}

@dataclass
class InputSpace:
    num_tokens: List[int] = field(default_factory=lambda: [1, 2, 3, 5, 10])
    output_mode: List[str] = field(default_factory=lambda: ['one_hot', 'multi_hot', 'count'])