import numpy as np
from dataclasses import dataclass
from typing import Union, Tuple, List, Optional

# 1. Define valid_test_case variable
valid_test_case = {
    'factor': 0.2,
    'value_range': (0, 255),
    'seed': 42,
    'inputs': np.random.uniform(low=0.0, high=255.0, size=(2, 2, 3)),
    'training': True
}

# 2. Parameters that can affect output shape: None
# RandomBrightness does not change tensor shape, only brightness values

@dataclass
class InputSpace:
    """Dataclass containing parameters that could affect output shape.
    Note: RandomBrightness does not change output shape,
    so these are all parameters from the function signature.
    """
    factor: List[Union[float, Tuple[float, float]]] = None
    value_range: List[Tuple[float, float]] = None
    seed: List[Optional[int]] = None
    training: List[bool] = None
    
    def __post_init__(self):
        if self.factor is None:
            # Continuous parameter discretized to 5 values including boundaries
            self.factor = [-1.0, -0.5, 0.0, 0.5, 1.0]
        
        if self.value_range is None:
            # 5 typical value ranges
            self.value_range = [
                (0, 255),    # Default
                (0, 1),      # Normalized
                (-1, 1),     # Negative to positive
                (0, 100),    # Custom range
                (0, 2)       # Small range
            ]
        
        if self.seed is None:
            # Discrete parameter with 5 values
            self.seed = [None, 0, 42, 100, 999]
        
        if self.training is None:
            # Discrete parameter with 2 values
            self.training = [True, False]