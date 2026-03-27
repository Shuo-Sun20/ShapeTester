import keras
import numpy as np

def call_func(inputs):
    return keras.ops.rsqrt(inputs)

random_tensor = keras.ops.convert_to_tensor(np.random.uniform(0.1, 10.0, size=(5,)))
example_output = call_func(random_tensor)