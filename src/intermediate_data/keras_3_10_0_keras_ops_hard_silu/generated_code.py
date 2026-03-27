import keras.ops as ops
import numpy as np

def call_func(inputs):
    return ops.hard_silu(x=inputs)

# Generate random input tensor
np.random.seed(42)
random_data = np.random.randn(2, 3).astype('float32')
input_tensor = ops.convert_to_tensor(random_data)

# Call function and save output
example_output = call_func(inputs=input_tensor)