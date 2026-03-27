import keras
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List

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
# Based on analysis: Only 'inputs' directly determines the output shape.
# The other parameters (name, dtype, trainable, autocast, activity_regularizer) 
# only affect metadata or training behavior, not the tensor shape.
# However, per requirements, we must include all parameters except 'inputs' 
# and provide discretized value spaces for each.

# Discretized value spaces for each parameter:
# - name: List of 5 string values
# - dtype: List of 5 dtype values (removed one from previous 6)
# - trainable: List of 2 boolean values
# - autocast: List of 2 boolean values  
# - activity_regularizer: List of 5 values (including None)

@dataclass
class InputSpace:
    """Contains all parameters that could affect output shape (except 'inputs')
    with discretized value ranges."""
    
    name: List[str] = field(default_factory=lambda: [
        'layer1', 'layer2', 'identity_layer', 'test_layer', 'final_layer'
    ])
    
    dtype: List[Optional[str]] = field(default_factory=lambda: [
        'float16', 'float32', 'float64', 'int32', None
    ])
    
    trainable: List[bool] = field(default_factory=lambda: [True, False])
    
    autocast: List[bool] = field(default_factory=lambda: [True, False])
    
    activity_regularizer: List[Optional[keras.regularizers.Regularizer]] = field(
        default_factory=lambda: [
            None,
            keras.regularizers.L1(0.01),
            keras.regularizers.L2(0.01),
            keras.regularizers.L1L2(l1=0.01, l2=0.01),
            keras.regularizers.L2(0.02)
        ]
    )