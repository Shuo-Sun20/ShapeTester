import keras
import numpy as np

def call_func(inputs):
    return keras.ops.relu6(inputs)

random_tensor = keras.ops.convert_to_tensor(np.random.randn(2, 3).astype(np.float32))
example_output = call_func(random_tensor)