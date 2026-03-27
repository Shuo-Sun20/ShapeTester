import keras
from dataclasses import dataclass, field

# 1. Define valid_test_case
valid_test_case = {
    'inputs': [keras.random.uniform((3, 4)), keras.random.uniform((3, 4))],
    'rtol': 1e-05,
    'atol': 1e-08,
    'equal_nan': False
}

# 2. Parameters affecting output shape (excluding "inputs"): None
# isclose output shape is determined solely by broadcast shape of x1 and x2 (via inputs parameter)
# rtol, atol, and equal_nan only affect values within the tensor, not its shape.

# 3. Value spaces for all call_func parameters (excluding inputs)
# Even though they don't affect shape, we'll create value spaces as requested:
# rtol: Continuous float, must be non-negative
rtol_values = [0.0, 1e-10, 1e-07, 1e-05, 1e-03, 1e-01, 1.0]
# atol: Continuous float, must be non-negative
atol_values = [0.0, 1e-12, 1e-10, 1e-08, 1e-06, 1e-04, 1e-02]
# equal_nan: Boolean discrete
equal_nan_values = [True, False]

# 4. Define InputSpace dataclass
@dataclass
class InputSpace:
    # Since no parameters affect shape beyond inputs, we include all parameters
    # with their value spaces as requested
    rtol: list = field(default_factory=lambda: [0.0, 1e-10, 1e-07, 1e-05, 1e-03, 1e-01, 1.0])
    atol: list = field(default_factory=lambda: [0.0, 1e-12, 1e-10, 1e-08, 1e-06, 1e-04, 1e-02])
    equal_nan: list = field(default_factory=lambda: [True, False])

# Test instantiation
var = InputSpace()