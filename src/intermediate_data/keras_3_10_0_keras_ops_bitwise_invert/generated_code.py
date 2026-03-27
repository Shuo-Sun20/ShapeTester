import keras
import numpy as np

def call_func(inputs):
    return keras.ops.bitwise_invert(x=inputs)

# Generate random integer tensor
np_random_array = np.random.randint(0, 100, size=(3, 4), dtype=np.int32)
random_tensor = keras.ops.convert_to_tensor(np_random_array)
example_output = call_func(inputs=random_tensor)