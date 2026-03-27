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

# 3. Value spaces for parameters (though they don't affect shape)
# factor: Continuous between -1.0 and 1.0, discretized
factor_values = [-1.0, -0.5, -0.2, 0.0, 0.2, 0.5, 1.0]

# value_range: Discrete - common ranges for image preprocessing
value_range_options = [
    (0, 255),
    (0.0, 1.0),
    (-1.0, 1.0),
    (0.0, 255.0),
    (0, 1)
]

# training: Discrete boolean
training_options = [True, False]

# seed: Discrete - common seed values including None
seed_options = [None, 42, 123, 2024, 9999]

# 4. Define InputSpace dataclass with all parameters
@dataclass
class InputSpace:
    factor: List[float] = None
    value_range: List[Tuple[Union[int, float], Union[int, float]]] = None
    seed: List[Optional[int]] = None
    training: List[bool] = None
    
    def __post_init__(self):
        if self.factor is None:
            self.factor = factor_values
        if self.value_range is None:
            self.value_range = value_range_options
        if self.seed is None:
            self.seed = seed_options
        if self.training is None:
            self.training = training_options

# Example instantiation
var = InputSpace()