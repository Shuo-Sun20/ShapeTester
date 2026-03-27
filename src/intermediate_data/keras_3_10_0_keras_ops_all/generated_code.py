import keras
import numpy as np

def call_func(inputs, axis=None, keepdims=False):
    return keras.ops.all(inputs, axis=axis, keepdims=keepdims)

# Generate random boolean tensor for input
np.random.seed(42)
random_bool_array = np.random.choice([True, False], size=(3, 4, 5))
x = keras.ops.convert_to_tensor(random_bool_array)

# Call the function and save output
example_output = call_func(inputs=x, axis=1, keepdims=True)