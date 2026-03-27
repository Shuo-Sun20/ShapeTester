import keras
from dataclasses import dataclass

def call_func(inputs):
    return keras.ops.blackman(inputs)

valid_test_case = {"inputs": keras.ops.convert_to_tensor(8)}

@dataclass
class InputSpace:
    inputs: list = None
    
    def __post_init__(self):
        if self.inputs is None:
            # Discrete values covering typical window lengths
            # Boundary: 1 (minimum valid), typical small/medium/large values
            self.inputs = [1, 2, 3, 5, 8]