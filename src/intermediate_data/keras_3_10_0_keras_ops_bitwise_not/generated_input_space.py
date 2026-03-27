import keras
import numpy as np
from dataclasses import dataclass

def call_func(inputs):
    return keras.ops.bitwise_not(inputs)

# Generate random integer tensor
np_input = np.random.randint(0, 256, size=(2, 3), dtype=np.int32)
input_tensor = keras.ops.convert_to_tensor(np_input, dtype='int32')
example_output = call_func(input_tensor)

valid_test_case = {'inputs': input_tensor}

@dataclass
class InputSpace:
    pass