from dataclasses import dataclass, field
import keras
import numpy as np

def call_func(inputs, bins=10, range=None):
    x = inputs[0]
    return keras.ops.histogram(x, bins=bins, range=range)

# 1. Valid test case
x = keras.ops.convert_to_tensor(np.random.rand(8))
valid_test_case = {
    "inputs": [x],
    "bins": 10,
    "range": None
}

# 2. & 3. Parameters affecting output shape: bins, range
# Discrete parameters: none in this case (bins is continuous integer space)
# Continuous parameters: bins (positive integer), range (tuple of floats or None)

# 4. InputSpace dataclass
@dataclass
class InputSpace:
    # bins: positive integer, affects output shape (length of histogram array)
    bins: list = field(default_factory=lambda: [1, 2, 5, 10, 20, 50, 100, 500, 1000])
    
    # range: affects bin edges but not the number of bins (shape remains same for fixed bins)
    # However, it affects the numerical range of output edges
    range: list = field(default_factory=lambda: [
        None,
        (0.0, 1.0),
        (0.0, 0.5),
        (0.2, 0.8),
        (-1.0, 1.0),
        (0.0, 100.0)
    ])

# Note: While range doesn't change the array length (determined by bins), 
# it affects the output tensor values and is included as requested