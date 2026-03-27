import keras
from dataclasses import dataclass, field
from typing import Optional

# 1. Define valid_test_case
start_tensor = keras.random.normal(shape=(1,))
stop_tensor = keras.random.normal(shape=(1,))

valid_test_case = {
    "inputs": [start_tensor, stop_tensor],
    "num": 50,
    "endpoint": True,
    "base": 10,
    "dtype": None,
    "axis": 0
}

# 2. Parameters affecting output shape (excluding "inputs"):
# - num: Directly determines number of samples along the specified axis
# - axis: Determines along which axis the samples are stored (relevant for array-like start/stop)
# - endpoint: Does NOT affect shape (always returns num samples)
# - base: Does NOT affect shape
# - dtype: Does NOT affect shape (only data type)

# 3. Value spaces for shape-affecting parameters:

# num: int, continuous but discrete (must be positive)
# Boundary values: 1, 0 (invalid but possible to test), typical values
num_values = [0, 1, 2, 10, 25, 50, 100, 1000]

# axis: int, continuous but discrete
# For array-like start/stop, axis can be negative or positive within tensor rank
axis_values = [-3, -2, -1, 0, 1, 2, 3]

# 4. Define InputSpace dataclass
@dataclass
class InputSpace:
    num: list = field(default_factory=lambda: [0, 1, 2, 10, 25, 50, 100, 1000])
    axis: list = field(default_factory=lambda: [-3, -2, -1, 0, 1, 2, 3])

# Example instantiation
var = InputSpace()