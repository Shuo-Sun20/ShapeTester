import keras
import numpy as np

def call_func(inputs, axis):
    return keras.ops.expand_dims(x=inputs, axis=axis)

# Generate random input tensor
random_tensor = keras.ops.convert_to_tensor(np.random.randn(3, 4))
# Call the function with valid axis
example_output = call_func(inputs=random_tensor, axis=1)