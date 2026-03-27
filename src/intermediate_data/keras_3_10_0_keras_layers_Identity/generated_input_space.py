import keras
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Any, List

# 1. Define valid_test_case
valid_test_case = {
    'inputs': keras.ops.convert_to_tensor(np.random.randn(2, 5).astype(np.float32)),
    'name': 'test_identity',
    'dtype': 'float32',
    'trainable': True,
    'autocast': True,
    'activity_regularizer': None
}

# 2. & 3. Parameters that can affect output shape and their value spaces
# Based on analysis: Only 'inputs' directly affects shape. Other parameters only affect
# metadata (dtype, trainable status, regularization) but not tensor shape.
# However, for completeness, we include all non-input parameters with their possible values.

@dataclass
class InputSpace:
    """
    Discretized value space for Identity layer parameters.
    Note: No parameters except 'inputs' affect output tensor shape.
    """
    # dtype: Can affect internal computation but not output shape
    dtype: List[Optional[str]] = field(
        default_factory=lambda: [
            None,           # Default behavior
            'float32',      # Common float type
            'float64',      # Higher precision
            'bfloat16',     # Lower precision
            'int32',        # Integer type
            'bool'          # Boolean type
        ]
    )
    
    # trainable: Affects gradient flow but not shape
    trainable: List[bool] = field(
        default_factory=lambda: [True, False]
    )
    
    # autocast: Affects automatic dtype casting but not shape
    autocast: List[bool] = field(
        default_factory=lambda: [True, False]
    )
    
    # activity_regularizer: Can affect training but not shape
    activity_regularizer: List[Optional[Any]] = field(
        default_factory=lambda: [
            None,
            keras.regularizers.L1L2(l1=0.01, l2=0.01),
            keras.regularizers.L1(0.01),
            keras.regularizers.L2(0.01)
        ]
    )