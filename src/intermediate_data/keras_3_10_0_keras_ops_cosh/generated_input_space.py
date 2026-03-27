import keras
from dataclasses import dataclass

# Task 1: Define valid_test_case
valid_test_case = {
    'inputs': keras.random.normal(shape=(3, 4))
}

# Task 2-4: Define InputSpace dataclass
# Note: call_func only has one parameter 'inputs' which affects output shape
# No other parameters exist beyond 'inputs'
@dataclass
class InputSpace:
    # No fields needed since no other parameters affect output shape
    pass

# Example usage to demonstrate it can be instantiated
var = InputSpace()