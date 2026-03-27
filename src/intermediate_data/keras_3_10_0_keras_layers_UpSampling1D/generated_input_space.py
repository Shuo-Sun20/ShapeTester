import keras
import numpy as np
from dataclasses import dataclass, field
from typing import List

# 1. Define valid_test_case
valid_test_case = {
    'size': 2,
    'inputs': np.random.randn(2, 3, 4).astype(np.float32)
}

# 2. Identify shape-affecting parameters (excluding inputs)
# Only 'size' parameter affects output shape

@dataclass
class InputSpace:
    size: List[int] = field(default_factory=lambda: [1, 2, 3, 4, 5])