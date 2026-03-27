import keras
import numpy as np

def call_func(function, inputs):
    return keras.ops.vectorized_map(function, inputs)

# Create a random tensor input
batch_size = 4
input_tensor = keras.ops.convert_to_tensor(
    np.random.randn(batch_size, 3).astype('float32')
)

# Define a simple function to vectorize
def square(x):
    return x ** 2

# Call the function and store output
example_output = call_func(square, input_tensor)