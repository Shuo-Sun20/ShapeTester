import keras
import numpy as np

def call_func(inputs):
    return keras.ops.erfinv(inputs)

random_tensor = np.random.uniform(-0.99, 0.99, (3, 4)).astype(np.float32)
example_output = call_func(random_tensor)