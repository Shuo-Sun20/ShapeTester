import tensorflow as tf
import numpy as np
from dataclasses import dataclass, field

# 1. Define valid_test_case dictionary
valid_test_case = {
    'inputs': tf.constant(np.random.randn(3, 4, 2).astype(np.float32)),
    'full_matrices': True,
    'name': None
}

# 2. Parameters affecting output shape (excluding inputs): full_matrices
# 3. Parameter analysis:
#    full_matrices: boolean (discrete) -> possible values: [True, False]

# 4. Define InputSpace dataclass
@dataclass
class InputSpace:
    full_matrices: list = field(default_factory=lambda: [True, False])