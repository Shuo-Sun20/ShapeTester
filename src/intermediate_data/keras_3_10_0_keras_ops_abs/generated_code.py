import keras
import numpy as np

def call_func(inputs):
    return keras.ops.abs(inputs)

# Generate random tensor input
random_tensor = keras.ops.convert_to_tensor(
    np.random.uniform(-10.0, 10.0, size=(3, 5))
)
example_output = call_func(random_tensor)